"""Turning an Opinion into a wire payload. The privacy boundary lives here.

PRD 5.4 and 6: reviewer_id is stored on the server and must not reach a
participant screen. That is enforced structurally rather than by discipline.

Three rules, and tests/test_payload_privacy.py asserts all three:

1. `participant_point` builds its dict by naming every key it emits. It never
   starts from `asdict(opinion)` and removes something. A denylist protects the
   fields you thought of when you wrote it and leaks every field added after --
   and the field most likely to be added later is exactly the kind that should
   not be broadcast.

2. There is no `include_reviewer=` flag, no `if admin:` branch, and no shared
   serialiser with a mode argument. A flag means one wrong call site leaks
   everything, and the wrong call site is usually the reconnect path nobody
   tests by hand.

3. The two functions are separate objects reached through separate channels
   (src/hub.py). A participant socket does not carry the field because nothing
   on its path can produce it, not because a condition happened to be false.
"""

from collections.abc import Iterable, Mapping, Sequence

import math

from src.models import Opinion
from src.roster import Roster

# What a participant may see. Also the contract test_payload_privacy.py pins.
#
# `neighbors` is on this list rather than the admin one on purpose. It is
# embedding-atlas's own column format -- {"ids": [...], "distances": [...]} -- and
# it relates opinions to opinions. Every id in it names a point already on the
# participant's screen, and the distances are between texts they can already read,
# so it carries nothing about authorship. The rule this file exists to enforce is
# about reviewer identity, not about the corpus being interconnected.
PARTICIPANT_KEYS = frozenset(
    {"id", "target_id", "text", "source", "week", "timestamp", "x", "y", "neighbors"}
)
ADMIN_KEYS = PARTICIPANT_KEYS | {"reviewer_id", "reviewer_name"}

# What an opinion with no computed neighbours carries. Present and empty rather
# than absent: embedding-atlas's viewer reads the column off every row, and a
# missing key makes the whole row's struct null in DuckDB rather than one field.
EMPTY_NEIGHBORS = {"ids": [], "distances": []}


def _finite(value) -> float:
    """A coordinate that is safe to put on the wire.

    JSON has no NaN or Infinity. Python's json.dumps emits them as bare `NaN` /
    `Infinity` anyway, and every browser's JSON.parse throws on both -- so a single
    non-finite coordinate does not corrupt one point, it makes the whole frame
    unparseable and takes the map down for everyone in the room at once.

    This is the last place before the socket, so it is the right place to be
    certain rather than hopeful. Upstream guards in src/projection.py should mean
    a non-finite value never arrives here; that is exactly why the cost of
    checking is worth paying, since a bug there would otherwise surface as every
    client going blank at the same moment.
    """
    f = float(value)
    return f if math.isfinite(f) else 0.0


def participant_point(
    op: Opinion, xy: Sequence[float] | None, nb: Mapping | None = None
) -> dict:
    """One map point as a participant may see it.

    Every key is written out below. If you add a field to Opinion, it does not
    appear here until somebody types it here -- which is the entire design.
    """
    x, y = (_finite(xy[0]), _finite(xy[1])) if xy is not None else (0.0, 0.0)
    return {
        "id": op.id,
        "target_id": op.target_id,
        "text": op.text,
        "source": op.source,
        "week": op.week,
        "timestamp": op.timestamp,
        "x": x,
        "y": y,
        # embedding-atlas's neighbours contract, verbatim: ids are row ids as given
        # by the viewer's id column, sorted nearest first, with the matching
        # distances alongside. Copied rather than referenced so a later mutation of
        # the shared dict cannot rewrite a frame already queued for the socket.
        "neighbors": {"ids": list(nb.get("ids", [])),
                      "distances": list(nb.get("distances", []))}
        if nb else dict(EMPTY_NEIGHBORS),
    }


def admin_point(op: Opinion, xy: Sequence[float] | None, roster: Roster,
                nb: Mapping | None = None) -> dict:
    """The same point, plus authorship, for the admin channel only (PRD 5.6).

    Same map, same data, wider exposure -- PRD 5.6 forbids a separate dataset,
    so this deliberately reuses the participant projection and adds to it rather
    than building a parallel record that could drift out of step.
    """
    entry = roster.resolve(op.reviewer_id)
    point = participant_point(op, xy, nb)
    point["reviewer_id"] = op.reviewer_id
    point["reviewer_name"] = entry.display_name if entry else op.reviewer_id
    return point


def participant_points(
    ops: Iterable[Opinion],
    coords: Mapping[str, Sequence[float]],
    neighbors: Mapping[str, Mapping] | None = None,
) -> list[dict]:
    nb = neighbors or {}
    return [participant_point(op, coords.get(op.id), nb.get(op.id)) for op in ops]


def admin_points(
    ops: Iterable[Opinion],
    coords: Mapping[str, Sequence[float]],
    roster: Roster,
    neighbors: Mapping[str, Mapping] | None = None,
) -> list[dict]:
    nb = neighbors or {}
    return [admin_point(op, coords.get(op.id), roster, nb.get(op.id)) for op in ops]
