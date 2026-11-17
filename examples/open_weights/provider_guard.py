"""Stop a hosted-model run at the provider's first error, warning or safety block.

A hosted run spends the user's own credit under the user's own account, and a provider may
restrict an account that keeps sending requests its safeguards decline. So the run fails closed:
no request is retried, and the first of the following ends it.

- An exception from a request: every HTTP error the Claude API returns (400 invalid request,
  401 authentication, 402 billing, 403 permission, 404 not found, 409 conflict, 413 too large,
  429 rate limit, 500 API error, 504 timeout, 529 overloaded, and an error event in a stream
  that Inspect cannot classify), a client timeout, a network failure, or a batch item that
  errored, expired or was canceled. Inspect raises each of these, or turns it into an output with
  the error set, which also halts. No request, batch creation or batch poll is retried.
- A stop reason other than a finished answer or the token cap: anything Inspect records as
  `unknown`, `model_length`, `tool_calls` or a content filter it cannot attribute to the model's
  safety classifier.
- A safety-classifier decline (stop reason `refusal`) once more than `max_blocks` have arrived.
- A warning from the Anthropic SDK or from Inspect's Anthropic provider or batcher, logged or
  raised as a Python warning, such as a deprecated model or a dropped parameter.

On a halt the guard records why in `halt.json` in the run folder (no prompt or answer text), and
cancels the run's batches still in progress at the provider: Inspect does not, and the provider
would keep processing and billing them. The runner cancels them again once Inspect returns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from inspect_ai.model import GenerateConfig, ModelOutput, get_model

# Loggers of the Anthropic SDK and of Inspect's Anthropic provider and batcher.
GUARDED_LOGGERS = (
    "anthropic",
    "inspect_ai.model._providers.anthropic",
    "inspect_ai.model._providers._anthropic_batch",
    "inspect_ai.model._providers.util.batch",
)
# Python warnings raised from those modules' files.
GUARDED_FILES = re.compile(
    r"[/\\](anthropic[/\\]|inspect_ai[/\\]model[/\\]_providers[/\\]"
    r"(anthropic|_anthropic_\w+|util[/\\]batch)\.py$)"
)
GUARDED_MODULES = (
    r"(anthropic|inspect_ai\.model\._providers\.(anthropic|_anthropic_\w+|util\.batch))"
)
ANSWERED = ("stop", "max_tokens")
# Batches created this long before the run started are left alone: they belong to something else.
CLOCK_SKEW = timedelta(minutes=5)
HALT_FILE = "halt.json"


class ProviderHalt(RuntimeError):
    """The provider reported a problem; the run stops instead of sending more requests."""


@dataclass
class ProviderGuard:
    output: Path
    max_blocks: int
    headers: dict[str, str] | None = None
    started: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    blocks: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    reason: str | None = None
    cancelled: list[str] = field(default_factory=list)
    _lock: asyncio.Lock | None = None

    async def before_request(self) -> None:
        """Send nothing once the run has halted or the provider has warned."""

        if self.reason is not None:
            raise ProviderHalt(self.reason)
        if self.warnings:
            await self.halt(f"provider warning: {self.warnings[0]}")

    async def check_output(self, output: ModelOutput) -> None:
        """Halt unless `output` is an answer, a truncated answer, or an allowed classifier block."""

        problem = self.problem(output)
        if problem is not None:
            await self.halt(problem)
        await self.before_request()

    def problem(self, output: ModelOutput) -> str | None:
        if not output.choices:
            return "the provider returned no choices"
        if output.error is not None:
            return f"the provider returned an error in place of an answer: {output.error}"
        reason = output.stop_reason
        if reason in ANSWERED:
            return None
        stop = output.choices[0].stop_details
        if reason == "content_filter" and stop is not None and stop.type == "refusal":
            category = stop.category or "unspecified"
            self.blocks[category] = self.blocks.get(category, 0) + 1
            total = sum(self.blocks.values())
            if total > self.max_blocks:
                return (
                    f"safety-classifier decline {total} (category {category}) exceeds the "
                    f"{self.max_blocks} allowed"
                )
            return None
        detail = f" ({stop.type}, {stop.category})" if stop is not None else ""
        return f"unexpected stop reason {reason}{detail}"

    async def halt(self, reason: str) -> None:
        """Record the first reason, cancel the run's batches, and raise."""

        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self.reason is None:
                self.reason = reason
                self.cancelled = await cancel_batches(self.started, self.headers)
                self.write()
        raise ProviderHalt(self.reason)

    def write(self) -> None:
        record = {
            "reason": self.reason,
            "halted_at": datetime.now(timezone.utc).isoformat(),
            "run_started_at": self.started.isoformat(),
            "classifier_blocks": self.blocks,
            "max_blocks": self.max_blocks,
            "warnings": self.warnings,
            "cancelled_batches": self.cancelled,
        }
        with (self.output / HALT_FILE).open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")

    # Logging and warnings

    def __enter__(self) -> ProviderGuard:
        global _ACTIVE
        if _ACTIVE is not None:
            raise RuntimeError("a provider guard is already active")
        self._handler = _RecordingHandler(self.warnings)
        self._levels: dict[str, int] = {}
        for name in GUARDED_LOGGERS:
            logger = logging.getLogger(name)
            self._levels[name] = logger.level
            if logger.getEffectiveLevel() > logging.WARNING:
                logger.setLevel(logging.WARNING)
            logger.addHandler(self._handler)
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.filterwarnings("always", module=GUARDED_MODULES)
        shown = warnings.showwarning

        def showwarning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: TextIO | None = None,
            line: str | None = None,
        ) -> None:
            if GUARDED_FILES.search(filename):
                self.warnings.append(f"{category.__name__}: {message}")
            shown(message, category, filename, lineno, file, line)

        warnings.showwarning = showwarning
        _ACTIVE = self
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        global _ACTIVE
        _ACTIVE = None
        self._warnings.__exit__(kind, error, traceback)
        for name in GUARDED_LOGGERS:
            logger = logging.getLogger(name)
            logger.removeHandler(self._handler)
            logger.setLevel(self._levels[name])


class _RecordingHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(f"{record.name}: {record.getMessage()}")


_ACTIVE: ProviderGuard | None = None


def active_guard() -> ProviderGuard:
    if _ACTIVE is None:
        raise ProviderHalt(
            "no provider guard is active: run hosted models through run_open_weights.py"
        )
    return _ACTIVE


def describe_error(error: BaseException) -> str:
    """The class, HTTP status, API error type and request id of an error, and its message.

    Inspect wraps a provider's error in a ModelGenerateError whose message is the whole request,
    prompt included. For such an error the provider's own error, its cause, is described
    instead, or its `provider_message` when there is no cause; the request never is.
    """

    wrapped = hasattr(error, "provider_message")
    source = error.__cause__ if wrapped and error.__cause__ is not None else error
    parts = [type(error).__name__] + ([type(source).__name__] if source is not error else [])
    for attribute in ("status_code", "type", "request_id"):
        value = getattr(source, attribute, None)
        if value is None:
            value = getattr(error, attribute, None)
        if value is not None and not callable(value):
            parts.append(f"{attribute}={value}")
    if source is error and wrapped:
        message = str(getattr(error, "provider_message", None) or "no provider message")
    else:
        message = str(source)
    return f"{' '.join(parts)}: {message}"


async def cancel_batches(started: datetime, headers: dict[str, str] | None = None) -> list[str]:
    """Cancel the Anthropic batches still in progress that were created since `started`.

    `headers` are the run's client headers, such as its workspace. Returns the ids cancelled; a
    failure to list or cancel is returned in their place, so that a halt is never hidden by the
    cleanup.
    """

    from anthropic import AsyncAnthropic

    since = started - CLOCK_SKEW
    cancelled: list[str] = []
    try:
        async with AsyncAnthropic(max_retries=0, default_headers=headers) as client:
            async for batch in client.messages.batches.list(limit=100):
                if batch.created_at >= since and batch.processing_status == "in_progress":
                    await client.messages.batches.cancel(batch.id)
                    cancelled.append(batch.id)
    except Exception as error:  # noqa: BLE001 - reported, never raised over the halt
        cancelled.append(f"cancelling failed: {describe_error(error)}")
    return cancelled


def preflight(model: str, config: GenerateConfig) -> dict[str, Any]:
    """Build the request parameters offline and refuse a configuration the API would warn about.

    No request is sent. Raises ProviderHalt on a deprecated model, a sampling parameter the
    current API rejects, a refusal fallback (which would measure another model), or a warning.
    """

    from anthropic.resources.messages.messages import DEPRECATED_MODELS

    name = model.split("/", 1)[1]
    if name in DEPRECATED_MODELS:
        raise ProviderHalt(f"{name} is deprecated ({DEPRECATED_MODELS[name]})")
    if config.fallback_models:
        raise ProviderHalt("refusal fallbacks would answer with another model")
    for parameter in ("temperature", "top_p", "top_k", "extra_body"):
        if getattr(config, parameter) is not None:
            raise ProviderHalt(f"{parameter} is set; the hosted configuration sends none")
    found: list[str] = []
    handler = _RecordingHandler(found)
    logger = logging.getLogger("inspect_ai.model._providers.anthropic")
    logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            api = get_model(model, api_key="preflight-no-request", memoize=False).api
            params, extra_body, _, _ = api.completion_config(config)  # type: ignore[attr-defined]
    finally:
        logger.removeHandler(handler)
    found += [
        str(caught_warning.message)
        for caught_warning in caught
        if GUARDED_FILES.search(caught_warning.filename)
    ]
    if found:
        raise ProviderHalt(f"the configuration draws a warning: {found[0]}")
    if extra_body:
        raise ProviderHalt(f"the request would carry extra fields: {sorted(extra_body)}")
    if params.get("output_config", {}).get("effort") != config.reasoning_effort:
        raise ProviderHalt(f"effort {config.reasoning_effort} is not what would be sent")
    return dict(params)
