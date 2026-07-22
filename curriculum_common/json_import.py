"""Bounded duplicate-key-safe JSON-object imports for classroom surfaces."""

from __future__ import annotations

import json
from typing import Any


def parse_json_object(data: bytes | str, *, max_bytes: int = 1_000_000) -> dict[str, Any]:
    """Parse one bounded JSON object while rejecting duplicates and non-finite values."""

    raw = data if isinstance(data, bytes) else data.encode("utf-8")
    if len(raw) > max_bytes:
        raise ValueError(f"JSON input exceeds the {max_bytes}-byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("JSON input must be UTF-8") from exc

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def _constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is not allowed: {value}")

    try:
        value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("input must contain a JSON object")
    return value


__all__ = ["parse_json_object"]
