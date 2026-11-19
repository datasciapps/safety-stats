"""Content-filter and token-limit stops read from Inspect logs, end to end.

The other tests of these options stand in for the stop reader. These analyse a real Inspect
log, held in memory with three of its outputs rewritten: one answer that ran out of tokens
without a score, one that a content filter stopped before any content, and one that ran out of
tokens and then failed with a sample error.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from inspect_safety_stats import analyze
from inspect_safety_stats.config import load_plan
from inspect_safety_stats.registry import ASSUMPTIONS

EXAMPLE = Path(__file__).parents[2] / "examples" / "real_simpleqa"
TRUNCATED, DECLINED, BROKEN = "5", "7", "9"


def _stop(sample: Any, reason: str, content: str) -> None:
    choice = sample.output.choices[0]
    choice.stop_reason = reason
    choice.message.content = content
    sample.output.completion = content
    sample.scores = {}


@pytest.fixture(scope="module")
def log() -> Any:
    from inspect_ai.log import EvalError, read_eval_log

    log = copy.deepcopy(read_eval_log(str(EXAMPLE / "simpleqa-8-real.eval")))
    samples = {str(sample.id): sample for sample in log.samples or []}
    _stop(samples[TRUNCATED], "max_tokens", "The series first aired on")
    _stop(samples[DECLINED], "content_filter", "")
    _stop(samples[BROKEN], "max_tokens", "The answer is")
    samples[BROKEN].error = EvalError(
        message="the provider closed the connection", traceback="", traceback_ansi=""
    )
    return log


def _analyze(log: Any, tmp_path: Path, **score: str) -> Any:
    plan = yaml.safe_load((EXAMPLE / "safety-plan.yaml").read_text(encoding="utf-8"))
    plan["inputs"]["design_manifest"] = (EXAMPLE / "design.csv").as_posix()
    plan["inputs"]["score"].update(missing="conservative_bounds", **score)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8", newline="\n")
    return analyze(plan=load_plan(path), logs=log)


def _counts(report: Any, prompt_id: str) -> tuple[int, int, int, int | None]:
    (item,) = [item for item in report.prompt_estimates if item.prompt_id == prompt_id]
    return (item.n_failure, item.n_unclassified, item.n_policy_blocked_safe, item.n_token_limit)


@pytest.mark.parametrize(
    ("token_limit", "content_filter_stops", "errors"),
    [("failure", "safe", "end_to_end"), ("safe", "failure", "count_as_failure")],
)
def test_both_stops_are_read_from_the_log_and_counted_as_the_plan_says(
    log: Any, tmp_path: Path, token_limit: str, content_filter_stops: str, errors: str
) -> None:
    report = _analyze(
        log,
        tmp_path,
        token_limit=token_limit,
        content_filter_stops=content_filter_stops,
        errors=errors,
    )
    by_failure = {"failure": 1, "safe": 0}
    # The unscored answer at the token limit counts as token_limit says.
    assert _counts(report, TRUNCATED) == (by_failure[token_limit], 0, 0, 1)
    # The output the content filter stopped counts as content_filter_stops says: safe is a
    # policy block, failure a refusal.
    blocked = content_filter_stops == "safe"
    assert _counts(report, DECLINED) == (by_failure[content_filter_stops], 0, int(blocked), 0)
    # A sample error comes first: the truncated answer that then failed follows score.errors,
    # unclassified under end_to_end (its stage unknown) and a failure under count_as_failure.
    failed = errors == "count_as_failure"
    assert _counts(report, BROKEN) == (int(failed), int(not failed), 0, 0)
    codes = [
        (item.code, item.scope.split("/prompts/")[1].split("/")[0])
        for item in report.integrity.diagnostics
        if "/prompts/" in (item.scope or "")
    ]
    assert ("TOKEN_LIMIT_REACHED", TRUNCATED) in codes
    assert ("TOKEN_LIMIT_REACHED", BROKEN) not in codes
    code = "TARGET_POLICY_BLOCKED_SAFE" if blocked else "TARGET_POLICY_BLOCKED_REFUSAL"
    assert (code, DECLINED) in codes
    # The report states the assumptions for the values the plan took.
    stated = {item.id: item.statement for item in report.assumptions_used}
    assert stated["token_limit"] == ASSUMPTIONS["token_limit"].stated_for(token_limit)[0]
    assert stated["content_filter_stops"] == (
        ASSUMPTIONS["content_filter_stops"].stated_for(content_filter_stops)[0]
    )


def test_at_the_defaults_both_stops_are_only_missing_scores(log: Any, tmp_path: Path) -> None:
    report = _analyze(log, tmp_path)
    assert _counts(report, TRUNCATED) == (0, 1, 0, None)
    assert _counts(report, DECLINED) == (0, 1, 0, None)
    assert not any(
        item.code in {"TOKEN_LIMIT_REACHED", "TARGET_POLICY_BLOCKED_SAFE"}
        for item in report.integrity.diagnostics
    )
