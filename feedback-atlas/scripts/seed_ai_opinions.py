#!/usr/bin/env python
"""Bulk-import AI-generated opinions straight into the store (PRD 4.3).

Writes to the database rather than through a websocket, for one reason: the
submit path is rate-limited to roughly one message a second per connection. That
limit is right for people and wrong for a script -- an instructor pasting thirty
model readings of a presentation would have two thirds of them refused, silently
from their point of view.

The server holds the corpus in memory, so a running instance will not notice
these rows by itself. Send it SIGUSR1 afterwards and it re-reads them and
broadcasts without dropping a single connection:

    uv run --python .venv/bin/python scripts/seed_ai_opinions.py \\
        --db atlas.db --roster roster.csv --reviewer instructor comments.csv
    kill -USR1 $(pgrep -f 'uvicorn.*src.server')

Input is CSV (target_id,text[,source][,week]) or JSONL with the same keys.
`source` defaults to ai, which is the point of the script and also PRD 5.3's
default; `week` defaults to --week.

Embeddings and coordinates are not computed here. The server backfills any text
it has not embedded on its next recompute, so the import stays a database write
and never needs the model loaded twice.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import SOURCES, WEEKS, new_opinion  # noqa: E402
from src.roster import Roster  # noqa: E402
from src.store import Store  # noqa: E402
from src.textnorm import normalize_text, text_hash  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").lstrip("﻿")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise SystemExit(f"{path} is empty")
    missing = {"target_id", "text"} - {(f or "").strip() for f in reader.fieldnames}
    if missing:
        raise SystemExit(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    return [dict(row) for row in reader]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help="CSV or JSONL of comments")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--reviewer", required=True,
                    help="roster id these are filed under (usually the instructor)")
    ap.add_argument("--week", type=int, default=1, choices=WEEKS)
    ap.add_argument("--source", default="ai", choices=SOURCES)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    roster = Roster.from_path(args.roster)
    reviewer = roster.resolve(args.reviewer)
    if reviewer is None:
        raise SystemExit(
            f"--reviewer {args.reviewer!r} is not on the roster. Authorship has to "
            "resolve to a real entry, or admin mode's author view files these "
            "under a name that does not exist.")
    targets = {t["id"] for t in roster.targets()}

    rows = read_rows(args.input)
    opinions, skipped = [], []
    for i, row in enumerate(rows, start=1):
        text = normalize_text(str(row.get("text") or ""))
        raw_target = str(row.get("target_id") or "").strip()
        entry = roster.resolve(raw_target) if raw_target else None
        target = entry.id if entry and entry.id in targets else None

        if not text:
            skipped.append((i, "empty text"))

            continue
        if len(text) > args.max_chars:
            skipped.append((i, f"longer than {args.max_chars} characters"))

            continue
        if target is None:
            # Refused, never guessed. Filing a model's reading against the wrong
            # project is invisible once it is on the map.
            skipped.append((i, f"unknown target {raw_target!r}"))

            continue

        source = str(row.get("source") or args.source).strip().lower()
        if source not in SOURCES:
            skipped.append((i, f"bad source {source!r}"))

            continue
        try:
            week = int(row.get("week") or args.week)
        except (TypeError, ValueError):
            skipped.append((i, f"bad week {row.get('week')!r}"))

            continue
        if week not in WEEKS:
            skipped.append((i, f"week {week} is not one of {WEEKS}"))

            continue

        opinions.append(new_opinion(reviewer_id=reviewer.id, target_id=target,
                                    text=text, source=source, week=week))

    for line, why in skipped:
        print(f"  skipped row {line}: {why}", file=sys.stderr)

    if args.dry_run:
        print(f"would import {len(opinions)} opinion(s), skipping {len(skipped)}")
        return 1 if skipped else 0

    store = Store(str(args.db))
    store.migrate()
    written = sum(store.insert_opinion(op, text_hash(op.text)) for op in opinions)
    store.close()

    print(f"imported {written} opinion(s) as {reviewer.id}"
          + (f", skipped {len(skipped)}" if skipped else ""))
    if written:
        print("send SIGUSR1 to a running server so it picks these up without a restart:")
        print("  kill -USR1 $(pgrep -f 'uvicorn.*src.server')")
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
