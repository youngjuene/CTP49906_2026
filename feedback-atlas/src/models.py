"""The Opinion record and the one function that decides what becomes one.

Two rules here carry more weight than they look like they do:

1. `reviewer_id` is never read from the submitted body. It comes from the
   connection's already-resolved roster entry. A client that sends
   {"reviewer_id": "someone.else"} is not rejected -- the key is simply not
   consulted -- so there is no code path where a submitted field decides
   authorship.

2. Validation blocks on exactly one thing: empty text (PRD 5.2). Everything else
   falls back to a default. This is a tool used *during* a presentation; a form
   that argues with someone mid-talk gets abandoned, and a missing week is worth
   less than a lost opinion.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Mapping

from src.roster import Roster, RosterEntry
from src.textnorm import normalize_text

WEEKS = (1, 2, 3, 4)


def _valid_week(value) -> bool:
    """True only for a real week number.

    `isinstance(True, int)` is True and `True == 1`, so a bare `week in WEEKS`
    accepts True and stores it. It then serialises as JSON `true`, and a client
    comparing `week === 1` silently never matches that point.
    """
    return not isinstance(value, bool) and value in WEEKS


SOURCES = ("ai", "human")
DEFAULT_SOURCE = "ai"  # PRD 5.3: recorded as AI unless the writer says otherwise


@dataclass(frozen=True)
class Opinion:
    id: str
    reviewer_id: str   # canonical roster id. Stored; never sent to participants.
    target_id: str
    text: str
    source: str        # "ai" | "human"
    week: int
    timestamp: str     # ISO-8601 UTC, 'Z'


def _now_iso() -> str:
    """ISO-8601 UTC to the millisecond.

    Milliseconds rather than seconds because the corpus is ordered by
    (timestamp, id) and that order is the row order of the projection matrix. At
    one-second resolution a burst of submissions during a presentation -- which is
    the normal case, not the edge case -- all tie, and the tiebreak is a random
    uuid, so arrival order is lost. Ordering stays deterministic either way; this
    just makes it also mean something.
    """
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_opinion(
    *,
    reviewer_id: str,
    target_id: str,
    text: str,
    source: str | None = None,
    week: int | None = None,
    now: str | None = None,
    default_week: int = 1,
) -> Opinion:
    return Opinion(
        id="o_" + uuid.uuid4().hex[:12],
        reviewer_id=reviewer_id,
        target_id=target_id,
        text=normalize_text(text),
        source=source if source in SOURCES else DEFAULT_SOURCE,
        week=week if _valid_week(week) else default_week,
        timestamp=now or _now_iso(),
    )


def validate_submission(
    body: Mapping,
    *,
    reviewer: RosterEntry,
    roster: Roster,
    max_chars: int = 1000,
    default_week: int = 1,
    now: str | None = None,
) -> tuple[Opinion | None, str | None]:
    """(opinion, None) on success, (None, error_code) on refusal.

    `reviewer` is the connection's identity. `body` is untrusted client input and
    its `reviewer_id`, if any, is ignored rather than honoured.
    """
    text = normalize_text(str(body.get("text") or ""))
    if not text:
        return None, "EMPTY_TEXT"
    if len(text) > max_chars:
        # Capped because every stored character rides in every future snapshot
        # and every reconnect, for the rest of the semester.
        return None, "TEXT_TOO_LONG"

    target_id = str(body.get("target_id") or "").strip()
    entry = roster.resolve(target_id) if target_id else None
    if entry is None or entry.role != "student":
        # Unlike week and source, this does not get a default: silently filing an
        # opinion against the wrong project is worse than refusing it.
        return None, "UNKNOWN_TARGET"

    raw_source = body.get("source")
    source = raw_source if raw_source in SOURCES else None
    if raw_source is not None and source is None:
        return None, "BAD_SOURCE"

    raw_week = body.get("week")
    week = None
    if raw_week is not None:
        if isinstance(raw_week, bool):
            # int(True) is 1, so without this a JSON `true` is quietly filed as
            # week 1. Every other malformed week is refused rather than guessed
            # at, and a boolean is no more a week than a 5 is.
            return None, "BAD_WEEK"
        try:
            week = int(raw_week)
        except (TypeError, ValueError):
            return None, "BAD_WEEK"
        if not _valid_week(week):
            # Refused, not clamped. Clamping a 5 to a 4 invents data that looks
            # exactly like data somebody entered.
            return None, "BAD_WEEK"

    return new_opinion(
        reviewer_id=reviewer.id,   # <- the connection, never body["reviewer_id"]
        target_id=entry.id,        # <- canonical, not as typed
        text=text,
        source=source,
        week=week,
        now=now,
        default_week=default_week,
    ), None
