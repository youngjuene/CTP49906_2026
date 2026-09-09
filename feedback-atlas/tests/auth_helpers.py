"""Test-only access-code fixtures for route authentication."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ACCESS_CODES = {
    "target1": "target-one-code",
    "target2": "target-two-code",
    "writer1": "writer-one-code",
    "writer2": "writer-two-code",
}

ACCESS_HASHES = {
    "target1": "9033c962a185589f0176c2501c5620f0491c6bc015c8ee6e9868e354a8451ee4",
    "target2": "331e4e9b819b5304af8685f39456dc0346e87c3f30d72111c79cb42dd5070f4a",
    "writer1": "147ba71ccdba6732e9c6c76b110db8128b07f3b5d074f8e0b5ffaf880c98322a",
    "writer2": "e7b3410771ebb413974102e2821451d7c0a854b209f6116557691affe8aa4ed2",
}


def write_access_codes(tmp_path: Path) -> Path:
    path = tmp_path / "access-codes.json"
    path.write_text(json.dumps(ACCESS_HASHES, indent=2, sort_keys=True), encoding="utf-8")
    return path


def code_for(identity: str) -> str:
    return ACCESS_CODES.get(identity, f"code-for-{identity}")


def write_access_codes_for(tmp_path: Path, identities) -> Path:
    path = tmp_path / "access-codes.json"
    hashes = {
        identity: hashlib.sha256(code_for(identity).encode("utf-8")).hexdigest()
        for identity in identities
    }
    path.write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
    return path
