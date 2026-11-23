from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any


def canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): canonical_value(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return canonical_value(value.value)
    if hasattr(value, "item"):
        return canonical_value(value.item())
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
