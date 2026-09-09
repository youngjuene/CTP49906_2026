"""Copy the current example corpus into demo storage and prepare an empty classroom.

Run with the classroom service stopped. The original database is retained as a
private backup. Existing destinations are never overwritten, and participant
credentials and the roster are left intact.
"""

import argparse
import contextlib
import os
import re
import sqlite3
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def separate(source: Path, demo: Path, classroom: Path, env_path: Path,
             expected_opinions: int) -> None:
    from src.store import Store
    import shlex

    paths = [p.resolve() for p in (source, demo, classroom)]
    if len(set(paths)) != 3 or not source.is_file():
        raise ValueError("Source, demo, and classroom must be separate database paths")
    for dest in (demo, classroom):
        if any(Path(str(dest) + suffix).exists() or Path(str(dest) + suffix).is_symlink()
               for suffix in ("", "-wal", "-shm")):
            raise FileExistsError("Destination already exists; refusing to overwrite feedback")

    # Validate configuration before creating any destination files.
    text = env_path.read_text()
    values = {"ATLAS_DB": str(classroom.resolve()), "ATLAS_DEMO_DB": str(demo.resolve()),
              "ATLAS_ENABLE_DEMO": "1"}
    for key, value in values.items():
        assignment = f"{key}={shlex.quote(value)}"
        pattern = rf"(?m)^(?:export\s+)?{key}=.*$"
        if re.search(pattern, text):
            text = re.sub(pattern, lambda _: assignment, text)
        else:
            text = text.rstrip() + "\n" + assignment + "\n"

    created = []
    temporary = None
    switched = False
    try:
        with contextlib.closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as original:
            count = original.execute("SELECT count(*) FROM opinions").fetchone()[0]
            if count != expected_opinions:
                raise ValueError(f"Expected {expected_opinions} example opinions, found {count}")
            for dest in (demo, classroom):
                dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
                created.append(dest)
            with contextlib.closing(sqlite3.connect(demo)) as copy:
                original.backup(copy)
                if copy.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise RuntimeError("Demo database integrity check failed")
                if copy.execute("SELECT count(*) FROM opinions").fetchone()[0] != count:
                    raise RuntimeError("Demo opinion count does not match the source")
        fresh = Store(str(classroom))
        try:
            fresh.migrate()
            if fresh.all_opinions():
                raise RuntimeError("New classroom database is not empty")
        finally:
            fresh.close()

        source.chmod(0o600)
        # Switch paths only after the complete copy and empty classroom are verified.
        fd, temporary = tempfile.mkstemp(prefix=".env-demo-", dir=env_path.parent)
        with os.fdopen(fd, "w") as out:
            out.write(text)
        os.replace(temporary, env_path)
        switched = True
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
        if not switched:
            # These paths were created exclusively above; the source is never removed.
            for dest in created:
                for suffix in ("", "-wal", "-shm"):
                    Path(str(dest) + suffix).unlink(missing_ok=True)
    print(f"Verified {count} demo opinions; classroom is empty. Original retained at {source}.")

def main():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--classroom", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--expected-opinions", type=int, required=True)
    args = parser.parse_args()
    separate(args.source, args.demo, args.classroom, args.env, args.expected_opinions)


if __name__ == "__main__":
    main()
