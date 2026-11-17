"""Run the open-weight demo against a vLLM server that is already running, then analyze it.

Only `vllm/`, `mockllm/` and `anthropic/` models are accepted, and the runner refuses to start
without `VLLM_BASE_URL`: the judge always runs on that server, and without it Inspect's vLLM
provider would start a server of its own. The API key is read by Inspect from `VLLM_API_KEY`;
never pass it on the command line. A local model samples as its card says, from the task's
`MODEL_CARDS`; a model without an entry there is refused, and `--reasoning-effort` is refused for
a model whose chat template takes none.

An `anthropic/` model is a hosted one, run in one configuration (`--arms on`, thinking is always
on) at `--reasoning-effort`, with batches at half price, and judged by the local model. Its key
is read from the file named by `--api-key-file` into `ANTHROPIC_API_KEY`; a key not scoped to a
workspace needs `--workspace-id`, sent on every request. The run fails closed
(see provider_guard.py): no request or sample is retried, the configuration is checked offline
first, and the provider's first error, warning or unexpected stop halts the run and cancels its
batches in progress.

Both thinking configurations run in one Inspect call, so their samples share the server's batch:
while one configuration drains its last long answers, the other keeps the batch full. The judge
calls go to the same server through the same connection pool. `--max-connections` is the number
of requests in flight at once, across both configurations and the judge, and it switches off
Inspect's adaptive connection control. Each configuration may hold that many samples at once, so
either can fill the pool alone.

Logs hold prompts and completions: the output folder must be outside the repository, and it is
created with mode 700.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.model import Model, get_model
from open_weights_task import (
    MODEL,
    REASONING_EFFORTS,
    XSTEST_ENV,
    card_for,
    design_rows,
    hosted_config,
    open_weights_safety_task,
)
from provider_guard import ProviderGuard, cancel_batches, preflight

HERE = Path(__file__).resolve().parent
LOCAL_PROVIDERS = ("vllm/", "mockllm/")
HOSTED_PROVIDERS = ("anthropic/",)
ARMS = {"off": False, "on": True}
REQUEST_TIMEOUT_ENV = "INSPECT_HTTP_REQUEST_TIMEOUT"
API_KEY_ENV = "ANTHROPIC_API_KEY"
# Each would send the hosted requests somewhere other than the key's own account.
FOREIGN_CREDENTIAL_ENV = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
HOSTED_MAX_TOKENS = 32000
# A key that is not scoped to a workspace must name one in this header on every request.
WORKSPACE_HEADER = "anthropic-workspace-id"
WORKSPACE_ID = re.compile(r"wrkspc_[A-Za-z0-9]+")
LOCAL_MAX_TOKENS_ON = 16384
DEFAULT_REASONING_EFFORT = "xhigh"


def check_model_name(name: str) -> str:
    if not name.startswith(LOCAL_PROVIDERS + HOSTED_PROVIDERS):
        raise SystemExit(f"{name}: only vllm/, mockllm/ and anthropic/ models are allowed")
    return name


def check_judge_name(name: str) -> str:
    if not name.startswith(LOCAL_PROVIDERS):
        raise SystemExit(f"{name}: the judge runs locally; only vllm/ and mockllm/ are allowed")
    return name


def is_hosted(name: str) -> bool:
    return name.startswith(HOSTED_PROVIDERS)


def load_api_key(
    key_file: Path | None, environ: dict[str, str] | os._Environ[str] = os.environ
) -> None:
    """Put the hosted model's key into its variable, from a file and never from the command line."""

    foreign = [name for name in FOREIGN_CREDENTIAL_ENV if environ.get(name)]
    if foreign:
        raise SystemExit(f"unset {', '.join(foreign)}: the run must use the key in --api-key-file")
    if key_file is None:
        raise SystemExit("a hosted model needs --api-key-file")
    key = key_file.expanduser().read_text(encoding="utf-8").strip()
    if not key or len(key.split()) != 1:
        raise SystemExit(f"{key_file}: expected one API key")
    environ[API_KEY_ENV] = key


def check_judge(name: str, environ: dict[str, str] | os._Environ[str] = os.environ) -> None:
    """Before a hosted model is paid for, make sure the local judge can be built and is served.

    A judge that fails ends a hosted run at its first score, after the provider has answered and
    billed every request already sent.
    """

    try:
        get_model(name, memoize=False)
    except Exception as error:
        raise SystemExit(f"the judge {name} cannot be used: {error}") from error
    url = environ["VLLM_BASE_URL"].rstrip("/") + "/models"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {environ.get('VLLM_API_KEY', '')}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            served = [entry["id"] for entry in json.load(response)["data"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"the judge's server at {url} does not answer: {error}") from error
    if name.split("/", 1)[1] not in served:
        raise SystemExit(f"the judge's server at {url} serves {served}, not {name}")


def check_environment(environ: dict[str, str] | os._Environ[str] = os.environ) -> None:
    if not environ.get("VLLM_BASE_URL"):
        raise SystemExit(
            "VLLM_BASE_URL is not set; point it at the running server (for example "
            "http://127.0.0.1:8000/v1) so that Inspect does not start one of its own"
        )


def check_output_dir(path: Path) -> Path:
    path = path.resolve()
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True, text=True
    )
    if top.returncode == 0 and path.is_relative_to(Path(top.stdout.strip()).resolve()):
        raise SystemExit(f"{path}: logs hold completions; write them outside the repository")
    return path


def select_rows(rows: list[dict[str, str]], per_category: int | None) -> list[dict[str, str]]:
    """All rows, or the first `per_category` prompt IDs (sorted) of every category."""

    if per_category is None:
        return rows
    by_category: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[(row["task_id"], row["category"])].append(row)
    return [
        row
        for key in sorted(by_category)
        for row in sorted(by_category[key], key=lambda r: r["prompt_id"])[:per_category]
    ]


def set_request_timeout(seconds: float) -> None:
    """Let one request run as long as a full batch needs.

    Requests are not streamed, so Inspect's HTTP client sees nothing until an answer is complete,
    and its default budget is 600 seconds. With 160 requests sharing the GPU each stream gets
    1.5-2 tokens/s, so a 1,000-token answer can exceed it: the request is dropped, its work is
    lost, and Inspect retries after an exponential backoff of up to 30 minutes. In an audit run
    on this node 22 requests timed out that way and left one request running for 14 minutes.
    """

    if seconds <= 0:
        raise SystemExit("--request-timeout must be positive")
    os.environ[REQUEST_TIMEOUT_ENV] = str(seconds)


def write_design(destination: Path, rows: list[dict[str, str]], epochs: int) -> None:
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[*rows[0], "expected_rollouts"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({**row, "expected_rollouts": str(epochs)} for row in rows)


def write_plan(
    destination: Path,
    *,
    run_id: str,
    model: str,
    intent: str,
    frozen_at: str,
    arms: list[str],
    reasoning_effort: str | None = None,
) -> None:
    """The template plan for the arms that run; a hosted model's title says so."""

    with (HERE / "plan-template.yaml").open(encoding="utf-8") as stream:
        plan: dict[str, Any] = yaml.safe_load(stream)
    plan["analysis"]["id"] = run_id
    plan["analysis"]["intent"] = intent
    plan["analysis"]["plan_frozen_at"] = frozen_at
    if is_hosted(model):
        plan["analysis"]["title"] = f"Hosted-model safety: {model}, effort {reasoning_effort}"
        plan["analysis"]["description"] = (
            "One hosted model in one configuration (it always thinks; the reasoning effort is "
            "fixed), on HarmBench's 300 standard and contextual text behaviors (DirectRequest; "
            "harmful compliance) and XSTest's 250 safe prompts (over-refusal), judged by a local "
            "open-weight model with each benchmark's own rubric. A rollout the provider's safety "
            "classifier declined before delivering any content is not judged: it counts as safe "
            "for harm and as a refusal for over-refusal."
        )
    for arm in ARMS:
        cohort = f"thinking_{arm}"
        if arm in arms:
            plan["cohorts"][cohort]["match"]["model"] = model
        else:
            del plan["cohorts"][cohort]
    if len(arms) < 2:
        del plan["comparisons"]
    plan["claims"] = [
        claim
        for claim in plan["claims"]
        if ("comparisons" in plan if "comparison" in claim else claim["cohort"] in plan["cohorts"])
    ]
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False)


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--judge-model", default=None, help=f"Defaults to --model, or {MODEL} for a hosted one."
    )
    parser.add_argument(
        "--judge-max-connections", type=int, default=None, help="Defaults to --max-connections."
    )
    parser.add_argument("--api-key-file", type=Path, default=None, help="Hosted models only.")
    parser.add_argument(
        "--workspace-id",
        default=None,
        help="Hosted models: the workspace to bill, for a key not scoped to one.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Hosted models: requests per batch, 0 for none (twice the price).",
    )
    parser.add_argument(
        "--max-provider-blocks",
        type=int,
        default=0,
        help="Hosted models: safety-classifier declines allowed before the run halts.",
    )
    parser.add_argument("--arms", default="off,on", help="Thinking settings: off, on or both.")
    parser.add_argument("--epochs", type=int, required=True, help="Rollouts per prompt.")
    parser.add_argument("--per-category", type=int, default=None, help="Pilot subset size.")
    parser.add_argument("--max-connections", type=int, required=True)
    parser.add_argument("--max-tokens-off", type=int, default=4096)
    parser.add_argument(
        "--max-tokens-on",
        type=int,
        default=None,
        help=f"Defaults to {LOCAL_MAX_TOKENS_ON}, or {HOSTED_MAX_TOKENS} for a hosted model.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=None,
        help=f"Defaults to {DEFAULT_REASONING_EFFORT} for a model that takes one.",
    )
    parser.add_argument("--intent", choices=["exploratory", "confirmatory"], default="exploratory")
    parser.add_argument("--xstest-csv", type=Path, default=None, help=f"Or set {XSTEST_ENV}.")
    parser.add_argument("--output", type=Path, required=True, help="New folder, outside the repo.")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=14400.0,
        help="Seconds one request may take (default: 4 hours). See set_request_timeout.",
    )
    parser.add_argument("--no-analyze", action="store_true")
    args = parser.parse_args(argv)
    args.arms = [arm.strip() for arm in args.arms.split(",")]
    if not args.arms or any(arm not in ARMS for arm in args.arms):
        parser.error("--arms takes off, on, or off,on")
    if args.epochs < 1 or args.max_connections < 1:
        parser.error("--epochs and --max-connections must be at least 1")
    if args.batch_size < 0 or args.max_provider_blocks < 0:
        parser.error("--batch-size and --max-provider-blocks cannot be negative")
    if args.workspace_id is not None and not WORKSPACE_ID.fullmatch(args.workspace_id):
        parser.error("--workspace-id looks like wrkspc_ followed by letters and digits")
    if args.max_tokens_on is None:
        args.max_tokens_on = HOSTED_MAX_TOKENS if is_hosted(args.model) else LOCAL_MAX_TOKENS_ON
    # A local model takes the efforts its card names; any other model is refused in run().
    efforts = REASONING_EFFORTS
    if args.model.startswith(LOCAL_PROVIDERS):
        try:
            efforts = card_for(args.model).reasoning_efforts
        except ValueError as error:
            parser.error(str(error))
    if args.reasoning_effort is None and efforts:
        args.reasoning_effort = DEFAULT_REASONING_EFFORT
    elif args.reasoning_effort is not None and not efforts:
        parser.error(f"{args.model} takes no --reasoning-effort")
    return args


def run(
    args: argparse.Namespace, model: Model | None = None, judge: Model | None = None
) -> list[Any]:
    """Generate and judge. `model` and `judge` replace the named models (tests only)."""

    check_environment()
    model_name = check_model_name(args.model)
    hosted = is_hosted(model_name)
    judge_name = check_judge_name(args.judge_model or (MODEL if hosted else args.model))
    if hosted:
        if args.arms != ["on"]:
            raise SystemExit(f"{model_name} always thinks: run it with --arms on")
        load_api_key(args.api_key_file)
        if judge is None:
            check_judge(judge_name)
        preflight(
            model_name,
            hosted_config(args.reasoning_effort, args.max_tokens_on, args.batch_size),
        )
    set_request_timeout(args.request_timeout)
    if args.xstest_csv is not None:
        os.environ[XSTEST_ENV] = str(args.xstest_csv.resolve())
    output = check_output_dir(args.output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)

    rows = select_rows(design_rows(), args.per_category)
    write_design(output / "design.csv", rows, args.epochs)
    timestamp = datetime.now(timezone.utc)
    write_plan(
        output / "plan.yaml",
        run_id=f"open-weights-{timestamp.strftime('%Y%m%dt%H%M%Sz')}",
        model=model_name,
        intent=args.intent,
        frozen_at=timestamp.isoformat(),
        arms=args.arms,
        reasoning_effort=args.reasoning_effort,
    )

    tasks = [
        open_weights_safety_task(
            thinking=ARMS[arm],
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens_on if ARMS[arm] else args.max_tokens_off,
            judge_model=judge if judge is not None else judge_name,  # type: ignore[arg-type]
            judge_max_connections=args.judge_max_connections or args.max_connections,
            hosted=hosted,
            batch_size=args.batch_size if hosted else 0,
            model_card=model_name,
        )
        for arm in args.arms
    ]
    print(f"Output: {output}")
    print(
        f"{len(rows)} prompts x {args.epochs} rollouts x {len(tasks)} configurations = "
        f"{len(rows) * args.epochs * len(tasks)} generations, each judged once"
    )
    evaluate: dict[str, Any] = dict(
        model=model if model is not None else model_name,
        epochs=args.epochs,
        sample_id=[row["prompt_id"] for row in rows],
        log_dir=str(output / "logs"),
        log_format="eval",
        max_connections=args.max_connections,
        max_samples=args.max_connections,
        max_tasks=len(tasks),
        display="plain",
        ctl_server=False,
    )
    if not hosted:
        logs = inspect_eval(tasks, retry_on_error=2, fail_on_error=False, **evaluate)
    else:
        # Fail closed: the SDK and Inspect retry nothing, and the first sample error ends the
        # evaluation, after the guard has recorded why and cancelled the batches in progress.
        headers = {WORKSPACE_HEADER: args.workspace_id} if args.workspace_id else None
        with ProviderGuard(output, args.max_provider_blocks, headers) as guard:
            try:
                logs = inspect_eval(
                    tasks,
                    model_args={"max_retries": 0, "default_headers": headers},
                    max_retries=0,
                    retry_on_error=0,
                    fail_on_error=True,
                    **evaluate,
                )
            finally:
                leftover = asyncio.run(cancel_batches(guard.started, headers))
                if leftover:
                    print(f"Cancelled at the provider after the run: {leftover}")
            if guard.reason is not None:
                raise SystemExit(f"halted: {guard.reason} (see {output / 'halt.json'})")
    if not logs or any(str(log.status) != "success" for log in logs):
        raise SystemExit("an Inspect evaluation did not finish successfully; read its log")
    return logs


def analyze_run(output: Path) -> str:
    from inspect_safety_stats import analyze

    report = analyze(plan=output / "plan.yaml", logs=output / "logs")
    report.write(output / "report")
    return str(report.decision)


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    run(args)
    if not args.no_analyze:
        print(f"Decision: {analyze_run(args.output.resolve())}")


if __name__ == "__main__":
    sys.exit(main())
