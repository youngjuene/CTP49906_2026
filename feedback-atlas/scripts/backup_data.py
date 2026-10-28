#!/usr/bin/env python
"""Snapshot the private data into a dated archive on another disk.

Everything the repository does not hold: .run/ (the classroom database, the
saved chats and their map, the admin code, TLS keys), roster.csv and .env.
SQLite files are copied with SQLite's own backup API rather than cp, so a
database the live server is writing to (WAL mode) is captured consistently;
each copy is integrity-checked before it goes into the archive.

    .venv/bin/python scripts/backup_data.py --dest ~/backups/feedback-atlas

Archives are 0600 in a 0700 directory, named feedback-atlas-<UTC time>.tar.gz,
and only the newest --keep are kept. They hold secrets and student data: copy
them off the machine only encrypted. To restore, stop the service and unpack an
archive over the project directory (see SECURITY_DEPLOYMENT.md).
"""

import argparse
import os
import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEMS = (".run", "roster.csv", ".env")
SKIP_SUFFIXES = ("-wal", "-shm", "-journal")    # captured through the backup API


def _snapshot_db(src: Path, dest: Path) -> None:
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(dest)
        try:
            source.backup(target)
            ok = target.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            target.close()
    finally:
        source.close()
    if ok != "ok":
        raise RuntimeError(f"{src}: integrity check failed on the copy: {ok}")


def backup(dest: Path, keep: int) -> Path:
    dest.mkdir(mode=0o700, parents=True, exist_ok=True)
    dest.chmod(0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final = dest / f"feedback-atlas-{stamp}.tar.gz"
    fd, tmp = tempfile.mkstemp(dir=dest, prefix=".partial-", suffix=".tar.gz")
    os.close(fd)
    os.chmod(tmp, 0o600)
    files = dbs = 0
    try:
        with tempfile.TemporaryDirectory(prefix="atlas-backup-") as work, \
                tarfile.open(tmp, "w:gz") as tar:
            for item in ITEMS:
                base = ROOT / item
                if not base.exists():
                    continue
                paths = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
                for path in paths:
                    rel = path.relative_to(ROOT)
                    if path.name.endswith(SKIP_SUFFIXES):
                        continue
                    if path.suffix == ".db":
                        copy = Path(work) / f"{dbs}.db"
                        _snapshot_db(path, copy)
                        tar.add(copy, arcname=str(rel))
                        dbs += 1
                    else:
                        tar.add(path, arcname=str(rel))
                    files += 1
        os.replace(tmp, final)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    old = sorted(dest.glob("feedback-atlas-*.tar.gz"))[:-keep] if keep > 0 else []
    for path in old:
        path.unlink()
    print(f"wrote {final} ({final.stat().st_size / 1e6:.1f} MB, {files} files, "
          f"{dbs} databases checked); removed {len(old)} old archive(s)")
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=Path.home() / "backups" / "feedback-atlas")
    ap.add_argument("--keep", type=int, default=14, help="newest archives to keep (default 14)")
    args = ap.parse_args(argv)
    backup(args.dest.expanduser(), args.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
