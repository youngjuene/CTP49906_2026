#!/usr/bin/env python
"""Build a pseudonymised archive of a finished semester (PRD 6, 7).

Writes two new files and touches the source database not at all:

    archive.csv   the opinions, with every id replaced by a random code
    mapping.csv   code -> real id and display name

The separation is the whole design. The archive is what you keep; the mapping is
what you destroy, or store somewhere the archive is not. Keeping both in one file
would be a rename, not a de-identification.

Codes are random and assigned in shuffled order, not derived from the id. A
derived code -- a hash of the id, or a sequential number in roster order -- is
reversible by anyone who can guess the scheme or knows the roster, which is
everyone the archive would ever be shared with.

    uv run --python .venv/bin/python scripts/deidentify.py \\
        --db atlas.db --roster roster.csv --out archive/ --targets

**The archive is pseudonymous, not anonymous.** The opinions are unchanged, and
people write things like "제가 발표에서 말했듯이" that identify them regardless of
what the id column says. PRD 11 is right that the real control for research use
is consent obtained separately from participation, and analysis only after grades
are final. This script is the substitution step, not the ethics.
"""

import argparse
import csv
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.roster import Roster  # noqa: E402
from src.store import Store  # noqa: E402


def assign_codes(ids, prefix: str, rng: secrets.SystemRandom) -> dict[str, str]:
    """One code per id, unpredictable and collision-free.

    Shuffled before assignment so the code order carries nothing -- neither
    roster order nor order of first appearance, both of which are recoverable
    by someone who has the class list.
    """
    shuffled = sorted(set(ids))
    rng.shuffle(shuffled)
    width = max(2, len(str(len(shuffled))))
    return {oid: f"{prefix}{i:0{width}d}" for i, oid in enumerate(shuffled, start=1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--targets", action="store_true",
                    help="also pseudonymise target_id (PRD 7 leaves this optional)")
    args = ap.parse_args(argv)

    if not args.db.exists():
        raise SystemExit(f"{args.db} does not exist")
    args.out.mkdir(parents=True, exist_ok=True)
    archive_path = args.out / "archive.csv"
    mapping_path = args.out / "mapping.csv"
    for path in (archive_path, mapping_path):
        if path.exists():
            raise SystemExit(
                f"{path} already exists. Refusing to overwrite: a second run "
                "would assign different codes, and the old mapping would then "
                "decode nothing.")

    store = Store(str(args.db))
    store.migrate()
    opinions = store.all_opinions()
    coords = store.all_coords()
    store.close()
    if not opinions:
        raise SystemExit(f"{args.db} holds no opinions")

    roster = Roster.from_path(args.roster)
    rng = secrets.SystemRandom()
    reviewers = assign_codes((o.reviewer_id for o in opinions), "R", rng)
    targets = (assign_codes((o.target_id for o in opinions), "P", rng)
               if args.targets else {})

    with archive_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "reviewer_code", "target", "text", "source",
                         "week", "timestamp", "x", "y"])
        for op in opinions:
            x, y = coords.get(op.id, ("", ""))
            writer.writerow([
                op.id, reviewers[op.reviewer_id],
                targets.get(op.target_id, op.target_id),
                op.text, op.source, op.week, op.timestamp, x, y,
            ])

    with mapping_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["code", "kind", "id", "display_name"])
        for oid, code in sorted(reviewers.items(), key=lambda kv: kv[1]):
            entry = roster.resolve(oid)
            writer.writerow([code, "reviewer", oid,
                             entry.display_name if entry else ""])
        for oid, code in sorted(targets.items(), key=lambda kv: kv[1]):
            entry = roster.resolve(oid)
            writer.writerow([code, "target", oid,
                             entry.display_name if entry else ""])

    print(f"archive : {archive_path}  ({len(opinions)} opinions, "
          f"{len(reviewers)} reviewers"
          + (f", {len(targets)} targets" if targets else "") + ")")
    print(f"mapping : {mapping_path}")
    print(f"source  : {args.db} unchanged")
    print()
    print("Keep these apart. The mapping is the only thing that re-identifies the")
    print("archive; destroy it, or store it where the archive is not.")
    print("The archive is pseudonymous, not anonymous -- the opinions still say")
    print("whatever their authors wrote about themselves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
