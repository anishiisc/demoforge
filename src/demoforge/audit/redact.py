"""Strip sensitive values from payloads before writing to the audit log."""

from __future__ import annotations

import os
import re
from typing import Any

# Patterns whose values should be redacted in logged data.
_KEY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"authorization", re.IGNORECASE),
    re.compile(r"anthropic[_-]?key", re.IGNORECASE),
]

# Also redact any value that looks like an Anthropic API key literal.
_VALUE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{10,}"),
]

_REDACTED = "[REDACTED]"


def _is_sensitive_key(key: str) -> bool:
    return any(p.search(key) for p in _KEY_PATTERNS)


def _redact_value(value: str) -> str:
    """Replace secret literals inside a string value."""
    for p in _VALUE_PATTERNS:
        value = p.sub(_REDACTED, value)
    return value


def redact(obj: Any, _depth: int = 0) -> Any:
    """Recursively redact sensitive data from a JSON-serialisable object.

    Returns a new object with sensitive fields replaced by ``[REDACTED]``.
    Limits recursion to avoid stack overflows on pathological inputs.
    """
    if _depth > 20:
        return obj
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and _is_sensitive_key(k):
                result[k] = _REDACTED
            else:
                result[k] = redact(v, _depth + 1)
        return result
    if isinstance(obj, list):
        return [redact(item, _depth + 1) for item in obj]
    if isinstance(obj, str):
        return _redact_value(obj)
    return obj


def redact_env_key(text: str) -> str:
    """Replace the live ANTHROPIC_API_KEY value if it appears in *text*."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key and key in text:
        text = text.replace(key, _REDACTED)
    return text
