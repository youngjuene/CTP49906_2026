#!/usr/bin/env python3
"""Generate private access codes and public credential hashes for a roster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import secrets
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.roster import Roster


def _sha256_hex(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _prepare_private_parent(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)


def _write_private_text(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    except Exception:
        try:
            path.unlink()
        finally:
            raise


def generate(roster_path: Path, hashes_path: Path, out_path: Path) -> tuple[int, Path, Path]:
    if hashes_path.exists():
        raise FileExistsError(f"{hashes_path} already exists; refusing to rotate credentials")
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists; refusing to rotate credentials")

    roster = Roster.from_path(roster_path)
    rows = []
    hashes = {}
    for entry in roster.entries():
        access_code = secrets.token_urlsafe(24)
        hashes[entry.id] = _sha256_hex(access_code)
        rows.append({
            "name": entry.display_name,
            "id": entry.id,
            "role": entry.role,
            "access_code": access_code,
        })

    _prepare_private_parent(hashes_path)
    _prepare_private_parent(out_path)

    hashes_text = json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_private_text(hashes_path, hashes_text)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["name", "id", "role", "access_code"])
    writer.writeheader()
    writer.writerows(rows)
    _write_private_text(out_path, buf.getvalue())

    return len(rows), hashes_path, out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--hashes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    count, hashes_path, out_path = generate(args.roster, args.hashes, args.out)
    print(f"generated {count} access codes")
    print(f"hashes: {hashes_path}")
    print(f"private csv: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
