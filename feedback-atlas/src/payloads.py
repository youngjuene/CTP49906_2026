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
PARTICIPANT_KEYS = frozenset(
    {"id", "target_id", "text", "source", "week", "timestamp", "x", "y"}
)
ADMIN_KEYS = PARTICIPANT_KEYS | {"reviewer_id", "reviewer_name"}


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


def participant_point(op: Opinion, xy: Sequence[float] | None) -> dict:
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
    }


def admin_point(op: Opinion, xy: Sequence[float] | None, roster: Roster) -> dict:
    """The same point, plus authorship, for the admin channel only (PRD 5.6).

    Same map, same data, wider exposure -- PRD 5.6 forbids a separate dataset,
    so this deliberately reuses the participant projection and adds to it rather
    than building a parallel record that could drift out of step.
    """
    entry = roster.resolve(op.reviewer_id)
    point = participant_point(op, xy)
    point["reviewer_id"] = op.reviewer_id
    point["reviewer_name"] = entry.display_name if entry else op.reviewer_id
    return point


def participant_points(
    ops: Iterable[Opinion], coords: Mapping[str, Sequence[float]]
) -> list[dict]:
    return [participant_point(op, coords.get(op.id)) for op in ops]


def admin_points(
    ops: Iterable[Opinion], coords: Mapping[str, Sequence[float]], roster: Roster
) -> list[dict]:
    return [admin_point(op, coords.get(op.id), roster) for op in ops]
