"""The Opinion record.

Opinions enter only through scripts/seed_ai_opinions.py, which builds them with
`new_opinion`: nothing a browser sends can create one.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

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
