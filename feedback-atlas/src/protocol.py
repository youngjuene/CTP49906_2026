"""Wire message construction, and the Korean text the participant actually reads.

Pure: builds and parses dicts, touches no socket. That is what lets
tests/test_protocol_messages.py assert the shape of every frame without a server.

Two things are deliberately absent from this protocol:

* **A week filter, in either direction.** Snapshots and deltas always carry all
  four weeks and the client hides the unchecked ones. This is structural, not a
  convention: if the server projected only the checked weeks, unchecking week 3
  would refit the layout and move every remaining point -- exactly what PRD 5.5
  forbids. There is no parameter to misuse.

* **Any client->server frame that writes.** Participants are anonymous viewers;
  opinions arrive only through scripts/seed_ai_opinions.py on the server.
"""

import json
import math
from typing import Any

PROTOCOL_VERSION = 1

ERROR_CODES = (
    "NOT_AUTHENTICATED", "BAD_ACCESS_CODE", "RATE_LIMITED",
    "TOO_LARGE", "MALFORMED", "PROTOCOL_MISMATCH", "UNKNOWN_OPINION", "NO_CHATS",
)

# One table, so the Korean the reader sees is stated once instead of scattered
# through the UI.
MESSAGES: dict[str, tuple[str, str]] = {
    "NOT_AUTHENTICATED": ("관리자만 쓸 수 있는 요청입니다.", "관리자 화면에서 다시 시도해 주세요."),
    "BAD_ACCESS_CODE": ("관리자 코드가 올바르지 않습니다.", ""),
    "RATE_LIMITED": ("너무 빠르게 보내고 있습니다.", "잠시 후 다시 시도해 주세요."),
    "TOO_LARGE": ("보낸 내용이 너무 큽니다.", "새로고침한 뒤 다시 시도해 주세요."),
    "MALFORMED": ("요청을 이해하지 못했습니다.", "새로고침한 뒤 다시 시도해 주세요."),
    "PROTOCOL_MISMATCH": ("앱이 오래된 버전입니다.", "새로고침해 주세요."),
    "UNKNOWN_OPINION": ("해당 의견을 찾을 수 없습니다.", "새로고침한 뒤 다시 시도해 주세요."),
    "NO_CHATS": ("아직 올라온 대화가 없습니다.", "대화 지도가 만들어진 뒤 다시 열어 주세요."),
}

MAX_FRAME_BYTES = 8192


def error(code: str, *, fatal: bool = False) -> dict:
    if code not in ERROR_CODES:
        raise KeyError(f"unknown error code {code!r}")
    message, hint = MESSAGES[code]
    return {"t": "error", "code": code, "message": message, "hint": hint, "fatal": fatal}


def hello_ok(
    *, channel: str, rev: int, layout_rev: int, targets: list[dict],
    roster: list[dict] | None = None,
) -> dict:
    """The handshake reply.

    The participant form carries target names only. Who wrote what -- the full
    roster -- is on the admin channel alone.
    """
    msg: dict[str, Any] = {
        "t": "hello_ok", "protocol": PROTOCOL_VERSION, "channel": channel,
        "rev": rev, "layout_rev": layout_rev, "targets": targets,
        "weeks": [1, 2, 3, 4], "sources": ["ai", "human"],
    }
    if roster is not None:      # admin channel only
        msg["roster"] = roster
    return msg


def snapshot(*, rev: int, layout_rev: int, points: list[dict]) -> dict:
    return {"t": "snapshot", "rev": rev, "layout_rev": layout_rev, "points": points}


def delta(
    *, from_rev: int, rev: int, layout_rev: int,
    added: list[dict], moved: list[dict], drift: float = 0.0,
) -> dict:
    """An incremental update.

    `from_rev` and `rev` are the revs of the *snapshot the layout was computed
    from*, not state.rev at broadcast time. Opinions that arrived mid-recompute
    have no coordinates yet, and reporting them as delivered would leave clients
    believing they hold a point the server cannot place.
    """
    return {
        "t": "delta", "from_rev": from_rev, "rev": rev, "layout_rev": layout_rev,
        # Same reason as payloads._finite: a NaN here makes the frame unparseable
        # for every client, not just this field.
        "added": added, "moved": moved,
        "drift": round(float(drift), 6) if math.isfinite(float(drift)) else 0.0,
    }


def pong() -> dict:
    return {"t": "pong"}


def neighbors(*, id: str, items: list[dict], ready: bool = True, nonce: str | None = None) -> dict:
    """Nearest opinions to one opinion, closest first.

    Shaped as {ids, distances} to match embedding-atlas's own `data.neighbors`
    contract, so a future move to their component needs no translation.

    `ready` separates "the layout has not been computed yet" from "this opinion
    genuinely has no neighbours". Both are an empty list, and they mean opposite
    things to whoever is looking at the screen.
    """
    msg = {
        "t": "neighbors", "id": id, "ready": ready,
        "ids": [i["id"] for i in items],
        "distances": [round(float(i["distance"]), 6) for i in items],
    }
    if nonce is not None:
        msg["nonce"] = nonce
    return msg


def parse_client_frame(raw: str | bytes, *, max_bytes: int = MAX_FRAME_BYTES):
    """-> (type, body) on success, or an error code string.

    Size is checked before parsing: a 10 MB frame should cost a length check,
    not a JSON parse.
    """
    payload = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(payload) > max_bytes:
        return "TOO_LARGE"
    try:
        body = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return "MALFORMED"
    if not isinstance(body, dict):
        return "MALFORMED"
    t = body.get("t")
    if not isinstance(t, str) or not t:
        return "MALFORMED"
    return t, body
