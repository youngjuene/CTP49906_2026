"""The pre-registered participant list, and the only gate into the app.

Registration is strict (PRD 5.1, closing the open question in PRD 11): an id that
is not on this list does not get in. There is deliberately **no** method here that
adds an entry at runtime -- no add(), no register(), no __setitem__. That absence
is the policy, and tests/test_roster.py asserts it, because the natural way to
handle a walk-in auditor mid-class is to quietly synthesise a roster row, and that
would turn a bounded set of writers into an unbounded one without anyone deciding
to.

Adding a late participant means editing the CSV and sending the process a SIGHUP
(src/server.py re-reads the file), which keeps every open websocket alive.
"""

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from src.textnorm import match_key, normalize_text

ROLES = ("student", "auditor")


@dataclass(frozen=True)
class RosterEntry:
    id: str            # canonical id, as written in the CSV
    display_name: str  # real name. Admin surfaces only (PRD 6).
    role: str          # "student" (수강생) | "auditor" (청강생)


class Roster:
    """An immutable id -> entry lookup with PRD 5.1's relaxed matching."""

    def __init__(self, entries: list[RosterEntry]):
        self._entries = list(entries)
        self._by_key: dict[str, RosterEntry] = {}
        for e in self._entries:
            key = match_key(e.id)
            if key in self._by_key:
                # Not a warning. Two rows whose ids differ only by case or
                # whitespace both answer to the same typed id, so one silently
                # shadows the other and one person's feedback is filed under
                # another person's name -- in the one view (admin) that exists to
                # attribute it correctly.
                raise ValueError(
                    f"roster has two entries matching {key!r} "
                    f"({self._by_key[key].id!r} and {e.id!r}); ids must be "
                    "distinct after trimming and case-folding"
                )
            self._by_key[key] = e

    @classmethod
    def from_csv_text(cls, text: str) -> "Roster":
        """Parse `id,display_name,role`. Tolerates a BOM, CRLF and stray spaces."""
        reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
        if reader.fieldnames is None:
            raise ValueError("roster CSV is empty; expected an id,display_name,role header")
        fields = {(f or "").strip().lower() for f in reader.fieldnames}
        missing = {"id", "display_name", "role"} - fields
        if missing:
            raise ValueError(
                f"roster CSV is missing column(s): {', '.join(sorted(missing))}"
            )

        entries: list[RosterEntry] = []
        for lineno, row in enumerate(reader, start=2):
            clean = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            rid = normalize_text(clean.get("id", ""))
            if not rid:
                continue  # blank line at the end of a hand-edited file
            role = clean.get("role", "").lower()
            if role not in ROLES:
                raise ValueError(
                    f"roster line {lineno}: role {role!r} is not one of {ROLES}"
                )
            entries.append(RosterEntry(
                id=rid,
                display_name=normalize_text(clean.get("display_name", "")) or rid,
                role=role,
            ))
        if not entries:
            raise ValueError(
                "roster has no entries; with strict registration that means "
                "nobody can get in, which is never what was meant"
            )
        return cls(entries)

    @classmethod
    def from_path(cls, path) -> "Roster":
        return cls.from_csv_text(Path(path).read_text(encoding="utf-8"))

    def resolve(self, raw_id: str) -> RosterEntry | None:
        """The typed id -> its canonical entry, or None if unlisted.

        Returning the *entry* rather than a bool is the point: callers store
        `entry.id`, never the string the person typed. Storing what was typed
        would file one person's opinions under three spellings and quietly break
        admin mode's author filter, which is the only reason admin mode exists.
        """
        return self._by_key.get(match_key(raw_id))

    def targets(self) -> list[dict]:
        """Students only -- the list of projects feedback can be *about*.

        Auditors write feedback but are not presenting, so they are not targets.
        """
        return [
            {"id": e.id, "display_name": e.display_name}
            for e in self._entries if e.role == "student"
        ]

    def entries(self) -> list[RosterEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, raw_id: object) -> bool:
        return isinstance(raw_id, str) and match_key(raw_id) in self._by_key
