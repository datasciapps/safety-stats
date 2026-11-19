"""Plan registration receipts. Experimental.

A receipt records a plan's canonical hash and the time it was registered. Its clock is the
local one, so a receipt proves nothing until its author anchors it somewhere a reader can
check: a signed commit, or a token from a timestamping service. ``analyze`` checks that the
receipt belongs to the plan and predates the earliest evaluation. It cannot check the anchor.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from inspect_safety_stats.config.models import StrictModel

RECEIPT_VERSION: Literal["1"] = "1"


class RegistrationReceipt(StrictModel):
    receipt_version: Literal["1"]
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registered_at: datetime
    package_version: str

    @field_validator("registered_at")
    @classmethod
    def has_a_time_zone(cls, value: datetime) -> datetime:
        # A naive time could be read in any zone, and so be moved before an evaluation.
        if value.tzinfo is None:
            raise ValueError("registered_at must carry a time zone")
        return value


def _rounded_up_to_the_second(moment: datetime) -> datetime:
    """``moment`` if it is a whole second, or else the next whole second."""

    whole = moment.replace(microsecond=0)
    return whole if whole == moment else whole + timedelta(seconds=1)


def write_receipt(
    path: Path, plan_sha256: str, *, package_version: str, now: datetime | None = None
) -> RegistrationReceipt:
    """Write a receipt for ``plan_sha256`` to ``path``, which must not exist yet.

    The receipt records ``now`` (by default the current time) in UTC, rounded up to the
    whole second.
    """

    registered_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    receipt = RegistrationReceipt(
        receipt_version=RECEIPT_VERSION,
        plan_sha256=plan_sha256,
        # Rounded up, never down: a receipt must not record a time before the registration.
        # Inspect records a run's start truncated to the second, so a truncated receipt
        # written within a second after the start would read as no later than it.
        registered_at=_rounded_up_to_the_second(registered_at),
        package_version=package_version,
    )
    text = json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    # Exclusive creation: overwriting a receipt would replace the earlier registration
    # time with a later one, and lose the record a reader may already have anchored.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return receipt


def load_receipt(path: str | Path) -> RegistrationReceipt:
    """Read a receipt; a malformed one raises ``ValueError`` naming the file."""

    try:
        return RegistrationReceipt.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"registration receipt {Path(path).as_posix()} is invalid: {exc}") from exc
