"""Wire message construction, and the Korean text the participant actually reads.

Pure: builds and parses dicts, touches no socket. That is what lets
tests/test_protocol_messages.py assert the shape of every frame without a server.

Two things are deliberately absent from this protocol:

* **A week filter, in either direction.** Snapshots and deltas always carry all
  four weeks and the client hides the unchecked ones. This is structural, not a
  convention: if the server projected only the checked weeks, unchecking week 3
  would refit the layout and move every remaining point -- exactly what PRD 5.5
  forbids. There is no parameter to misuse.

* **A `reviewer_id` field on any client->server frame.** Authorship comes from the
  connection. See src/models.validate_submission.
"""

import json
import math
from typing import Any

PROTOCOL_VERSION = 1

ERROR_CODES = (
    "UNKNOWN_ID", "EMPTY_TEXT", "TEXT_TOO_LONG", "UNKNOWN_TARGET", "BAD_WEEK",
    "BAD_SOURCE", "NOT_AUTHENTICATED", "BAD_ACCESS_CODE", "RATE_LIMITED",
    "TOO_LARGE", "MALFORMED", "PROTOCOL_MISMATCH", "UNKNOWN_OPINION", "DEMO_FULL", "OUTSIDE_CLASS_PERIOD",
    "STALE_SCHEDULE", "BAD_SCHEDULE",
)

# One table, so PRD 5.1's escalation -- suspect a typo first, send them to the
# instructor only after that -- is stated once instead of scattered through the UI.
MESSAGES: dict[str, tuple[str, str]] = {
    "STALE_SCHEDULE": ("다른 관리자가 일정을 변경했습니다.", "현재 일정을 다시 불러온 뒤 수정해 주세요."),
    "BAD_SCHEDULE": ("수업 일정을 확인해 주세요.", "네 주차의 시작일을 순서대로 지정하고, 7일 기간이 서로 겹치지 않게 해 주세요."),
    "OUTSIDE_CLASS_PERIOD": ("현재는 수업 의견 접수 기간이 아닙니다.", "안내된 수업 일정을 확인해 주세요. 작성 중인 의견은 그대로 남아 있습니다."),
    "DEMO_FULL": ("데모 연습 공간이 가득 찼습니다.", "조교에게 연습 데이터 초기화를 요청해 주세요."),
    "UNKNOWN_ID": ("명단에서 찾을 수 없습니다.",
                   "오타가 없는지 먼저 확인해 주세요. 여러 번 실패하면 강사·조교에게 등록을 요청해 주세요."),
    "EMPTY_TEXT": ("의견을 입력해 주세요.", "내용이 있어야 지도에 올릴 수 있습니다."),
    "TEXT_TOO_LONG": ("의견이 너무 깁니다.", "1000자 이내로 줄여 주세요."),
    "UNKNOWN_TARGET": ("대상 학생을 찾을 수 없습니다.", "목록에서 다시 골라 주세요."),
    "BAD_WEEK": ("주차 값이 올바르지 않습니다.", "1~4주차 중에서 골라 주세요."),
    "BAD_SOURCE": ("구분 값이 올바르지 않습니다.", "AI 또는 사람 중에서 골라 주세요."),
    "NOT_AUTHENTICATED": ("먼저 ID를 확인해 주세요.", "새로고침한 뒤 다시 들어와 주세요."),
    "BAD_ACCESS_CODE": ("접근 코드가 올바르지 않습니다.", ""),
    "RATE_LIMITED": ("너무 빠르게 보내고 있습니다.", "잠시 후 다시 시도해 주세요."),
    "TOO_LARGE": ("보낸 내용이 너무 큽니다.", "의견을 줄여서 다시 보내 주세요."),
    "MALFORMED": ("요청을 이해하지 못했습니다.", "새로고침한 뒤 다시 시도해 주세요."),
    "PROTOCOL_MISMATCH": ("앱이 오래된 버전입니다.", "새로고침해 주세요."),
    "UNKNOWN_OPINION": ("해당 의견을 찾을 수 없습니다.", "새로고침한 뒤 다시 시도해 주세요."),
}

MAX_FRAME_BYTES = 8192


def error(code: str, *, fatal: bool = False) -> dict:
    if code not in ERROR_CODES:
        raise KeyError(f"unknown error code {code!r}")
    message, hint = MESSAGES[code]
    return {"t": "error", "code": code, "message": message, "hint": hint, "fatal": fatal}


def hello_ok(
    *, channel: str, rev: int, layout_rev: int, targets: list[dict],
    role: str | None = None, roster: list[dict] | None = None,
    default_source: str = "ai", default_week: int = 1, max_text_chars: int = 1000,
) -> dict:
    """The handshake reply.

    Note what the participant form is told: `role`, never a display name. The
    participant channel never carries a real name at all, not even the viewer's
    own -- there is nothing on this channel to correlate against.
    """
    msg: dict[str, Any] = {
        "t": "hello_ok", "protocol": PROTOCOL_VERSION, "channel": channel,
        "rev": rev, "layout_rev": layout_rev, "targets": targets,
        "weeks": [1, 2, 3, 4], "sources": ["ai", "human"],
        "default_source": default_source, "default_week": default_week,
        "max_text_chars": max_text_chars,
    }
    if role is not None:
        msg["role"] = role
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


def ack(*, nonce: str, id: str, rev: int) -> dict:
    """Sent to the submitter alone, before the delta. Their text landed."""
    return {"t": "ack", "nonce": nonce, "id": id, "rev": rev}


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
