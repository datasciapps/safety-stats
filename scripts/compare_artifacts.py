"""Show that a behaviour-neutral change leaves every report artifact byte-identical.

A class A change, such as a refactor or a performance change, must not change a single output
byte. This runs ``safety-stats analyze`` on the bundled examples, on a synthetic stress
design, and on offline cases built for the paths the examples do not reach (two cohorts and
comparisons, every error stage, judge panels and calibration, sampled designs, an incomplete run,
and ``generative analyze``; see ``_artifact_cases.py``), twice: with the package source as
committed at --base (default HEAD, so the change under test is the uncommitted one), and with the
working tree or the commit --new names. Every artifact, the command's output and its exit status
must match exactly.

    python scripts/compare_artifacts.py
    python scripts/compare_artifacts.py --base main --case synthetic --prompts 1000
    python scripts/compare_artifacts.py --case errors --case generative

The working tree is never modified: committed source is exported to a temporary directory and
imported from there. Both runs read the same inputs and write to the same output path. When
anything differs, the outputs are kept for inspection. Exit status: 0 identical, 1 different,
2 could not run, which includes a case whose base run wrote no artifacts.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from _artifact_cases import BUILDERS, Builder, build_case
from _source_trees import (
    WRONG_SOURCE,
    export_source,
    repository_root,
    resolve_commit,
    run_module,
    source_changed,
)

CASES = ("simpleqa", "xstest", "synthetic", *BUILDERS)


@dataclass(frozen=True)
class Case:
    name: str
    plan: Path
    logs: Path
    command: tuple[str, ...] = ("analyze",)
    extra: tuple[str, ...] = ()


@dataclass(frozen=True)
class Run:
    exit_status: int
    stdout: str
    stderr: str
    seconds: float


def bundled_cases(repository: Path) -> dict[str, Case]:
    """The examples that ship real Inspect logs."""
    simpleqa = repository / "examples" / "real_simpleqa"
    xstest = repository / "examples" / "real_xstest"
    return {
        "simpleqa": Case(
            "simpleqa", simpleqa / "safety-plan.yaml", simpleqa / "simpleqa-8-real.eval"
        ),
        "xstest": Case("xstest", xstest / "safety-plan.yaml", xstest / "current-log"),
    }


def synthetic_case(repository: Path, directory: Path, prompts: int) -> Case:
    """A larger design cloned from the bundled SimpleQA log, written under ``directory``.

    Prompts sit two to a question and spread over three tasks. Each has three epochs whose
    scores come from the eight real samples, so prompts differ in failure rate. The plan is
    the SimpleQA plan with questions as the primary independence blocks, which exercises the
    block index, the hierarchy nodes, the calibration and the region projection at a size the
    eight-prompt example cannot reach.
    """
    import yaml
    from inspect_ai.log import read_eval_log, write_eval_log

    template = repository / "examples" / "real_simpleqa"
    log = read_eval_log(str(template / "simpleqa-8-real.eval"))
    originals = log.samples or []
    epochs, tasks = 3, 3
    samples = []
    for index in range(prompts):
        for epoch in range(1, epochs + 1):
            sample = originals[(3 * index + epoch) % len(originals)].model_copy(deep=True)
            sample.id = f"p{index:05d}"
            sample.epoch = epoch
            sample.uuid = f"synthetic-{index:05d}-{epoch}"
            samples.append(sample)
    log.samples = samples
    log.eval.config.epochs = epochs
    if log.results is not None:
        log.results.total_samples = len(samples)
    (directory / "logs").mkdir(parents=True)
    write_eval_log(log, str(directory / "logs" / "synthetic.eval"))

    rows = ["prompt_id,question_id,task_id,expected_rollouts,zero_tolerance,mandatory"]
    rows += [
        f"p{index:05d},q{index // 2:05d},task-{(index // 2) % tasks},{epochs},false,true"
        for index in range(prompts)
    ]
    (directory / "design.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")

    plan = yaml.safe_load((template / "safety-plan.yaml").read_text(encoding="utf-8"))
    plan["analysis"]["id"] = "synthetic-stress"
    plan["analysis"]["title"] = f"Synthetic stress design: {prompts} prompts, {epochs} epochs"
    plan["groups"] = [
        {"id": "all", "title": "All prompts", "filter": {}, "weights": {"strategy": "equal_prompt"}}
    ]
    plan["design"]["independence_profiles"]["question"] = {
        "block_by": ["question_id"],
        "comparison_coupling": "independent",
    }
    direct = plan["outputs"]["direct_mean_intervals"]
    direct["groups"] = ["all"]
    direct["independence"]["primary"] = "question"
    for claim in plan["claims"]:
        claim["group"] = "all"
    (directory / "plan.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False), encoding="utf-8", newline="\n"
    )
    return Case("synthetic", directory / "plan.yaml", directory / "logs" / "synthetic.eval")


def _files(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }


def differing_files(base: Path, new: Path) -> list[str]:
    """Files, relative to the two directories, that differ in bytes or exist on one side only."""
    return sorted(
        name
        for name in _files(base) | _files(new)
        if not (base / name).is_file()
        or not (new / name).is_file()
        or (base / name).read_bytes() != (new / name).read_bytes()
    )


# Artifacts such as hierarchy-risks.csv hold whole JSON documents on each row.
_EXCERPT_WIDTH = 160


def _shorten(line: str) -> str:
    if len(line) <= _EXCERPT_WIDTH:
        return line
    return f"{line[:_EXCERPT_WIDTH]} ... [{len(line) - _EXCERPT_WIDTH} more characters]"


def _excerpt(base: Path, new: Path, limit: int = 30) -> str:
    """The start of a unified diff between two text files."""
    try:
        before = base.read_text(encoding="utf-8").splitlines() if base.is_file() else []
        after = new.read_text(encoding="utf-8").splitlines() if new.is_file() else []
    except UnicodeDecodeError:
        return "    (binary content differs)"
    lines = list(difflib.unified_diff(before, after, "base", "new", lineterm="", n=1))
    shown = [f"    {_shorten(line)}" for line in lines[:limit]]
    if len(lines) > limit:
        shown.append(f"    ... and {len(lines) - limit} more diff lines")
    return "\n".join(shown)


def _fail(message: str) -> int:
    print(f"compare_artifacts: {message}", file=sys.stderr)
    return 2


def _analyze(source: Path, hidden: Path | None, case: Case, output: Path) -> Run:
    arguments = [*case.command, "--plan", str(case.plan), "--logs", str(case.logs), *case.extra]
    started = time.perf_counter()
    completed = run_module(
        source,
        "safety_stats.cli",
        [*arguments, "--output", str(output)],
        hidden=hidden,
        capture=True,
        # Inspect drops the drive letter of logs on another Windows drive than the working
        # directory (#35). Both sides would then fail alike, which compares as identical.
        cwd=case.logs.parent,
    )
    seconds = time.perf_counter() - started
    return Run(completed.returncode, completed.stdout, completed.stderr, seconds)


def _compare_case(scratch: Path, name: str, runs: dict[str, Run]) -> list[str]:
    """Print how one case compares and return what differs."""
    base_dir, new_dir = scratch / "base" / name, scratch / "new" / name
    files = differing_files(base_dir, new_dir)
    base, new = runs["base"], runs["new"]
    problems = [f"{name}/{file}" for file in files]
    if base.exit_status != new.exit_status:
        problems.append(f"exit status: {base.exit_status} at base, {new.exit_status} new")
    if base.stdout != new.stdout:
        problems.append("the command's output")
    timing = f"base {base.seconds:.1f} s, new {new.seconds:.1f} s, one run each"
    if not problems:
        count = len(_files(new_dir))
        print(f"{name}: identical, {count} files, exit status {new.exit_status} ({timing})")
        return []
    print(f"{name}: DIFFERENT ({timing})")
    for problem in problems:
        print(f"  {problem}")
    # checksums.json changes whenever another artifact does; show the artifact itself.
    for file in sorted(files, key=lambda file: file == "checksums.json")[:1]:
        print(_excerpt(base_dir / file, new_dir / file))
    for side, run in runs.items():
        if run.exit_status != 0 and run.stderr.strip():
            tail = run.stderr.strip().splitlines()[-10:]
            print(f"  stderr at {side}:\n    " + "\n    ".join(tail))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show that report artifacts are byte-identical before and after a change.",
        epilog="Exit status: 0 identical, 1 different, 2 could not run.",
        allow_abbrev=False,
    )
    parser.add_argument("--base", default="HEAD", help="commit to compare against (default: HEAD)")
    parser.add_argument("--new", help="commit to compare instead of the working tree")
    parser.add_argument("--src", default="src", help="source directory (default: src)")
    parser.add_argument(
        "--case", action="append", choices=CASES, help="case to run, repeatable (default: all)"
    )
    parser.add_argument(
        "--prompts", type=int, default=240, help="prompts in the synthetic design (default: 240)"
    )
    parser.add_argument("--keep", action="store_true", help="keep the outputs when identical")
    options = parser.parse_args(argv)
    if options.prompts < 2:
        parser.error("--prompts must be at least 2")

    repository = repository_root()
    if repository is None:
        return _fail("run this inside a git checkout")
    base = resolve_commit(repository, options.base)
    if base is None:
        return _fail(f"{options.base!r} is not a commit")
    new = None if options.new is None else resolve_commit(repository, options.new)
    if options.new is not None and new is None:
        return _fail(f"{options.new!r} is not a commit")
    working_source = repository / options.src
    if new is None and not source_changed(repository, base, options.src):
        return _fail(
            f"{options.src}/ is the same at {options.base} and in the working tree, so there is "
            "nothing to compare. If the change is committed, pass --base with the commit "
            "before it."
        )
    base_label = f"{options.base} ({base[:10]})"
    new_label = "the working tree" if new is None else f"{options.new} ({new[:10]})"

    scratch = Path(tempfile.mkdtemp(prefix="compare_artifacts-"))
    keep = options.keep
    try:
        base_source = export_source(repository, base, options.src, scratch / "base-source")
        new_source = (
            working_source
            if new is None
            else export_source(repository, new, options.src, scratch / "new-source")
        )
        if base_source is None or new_source is None:
            return _fail(f"{options.src}/ does not exist at one of the commits")
        # The exported trees run with the working tree's source hidden from sys.path.
        sides = {
            "base": (base_source, working_source),
            "new": (new_source, None if new is None else working_source),
        }
        names = options.case or list(CASES)
        cases = bundled_cases(repository)
        if "synthetic" in names:
            cases["synthetic"] = synthetic_case(repository, scratch / "inputs", options.prompts)
        # Built with the base source where the package takes part, so inputs are the same for
        # both sides whatever the change under test does.
        builder = Builder(repository, scratch / "inputs", base_source, working_source)
        for name in names:
            if name in BUILDERS:
                try:
                    cases[name] = Case(name, *build_case(name, builder))
                except Exception as error:  # an input that cannot be built: nothing to compare
                    return _fail(f"could not build the {name} case: {error}")

        different: list[str] = []
        for name in names:
            runs: dict[str, Run] = {}
            for side, (source, hidden) in sides.items():
                # Both sides write to the same path, so an artifact that records it still matches.
                output = scratch / "run" / name
                runs[side] = _analyze(source, hidden, cases[name], output)
                if runs[side].exit_status == WRONG_SOURCE:
                    print(runs[side].stderr, file=sys.stderr)
                    return _fail("the package did not import from the tree under test")
                # A base run that writes nothing failed before reporting. The new side failing
                # the same way would match it in every byte and prove nothing.
                if side == "base" and not _files(output):
                    tail = "\n    ".join(runs[side].stderr.strip().splitlines()[-10:])
                    return _fail(
                        f"the {name} case wrote no artifacts at base (exit status "
                        f"{runs[side].exit_status}), so there is nothing to compare."
                        + (f"\n  stderr at base:\n    {tail}" if tail else "")
                    )
                (scratch / side).mkdir(exist_ok=True)
                if output.exists():
                    output.rename(scratch / side / name)
            different += _compare_case(scratch, name, runs)

        if different:
            keep = True
            print(f"DIFFERENT: the output changes against {base_label}. Outputs kept in {scratch}")
            return 1
        print(f"IDENTICAL: every artifact, output and exit status matches between {base_label} "
              f"and {new_label}.")
        if keep:
            print(f"Outputs kept in {scratch}")
        return 0
    finally:
        if not keep:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
