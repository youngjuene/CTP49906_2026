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

PROTOCOL_VERSION = 2

ERROR_CODES = (
    "UNKNOWN_ID", "EMPTY_TEXT", "TEXT_TOO_LONG", "UNKNOWN_TARGET", "BAD_WEEK",
    "BAD_SOURCE", "NOT_AUTHENTICATED", "BAD_ACCESS_CODE", "RATE_LIMITED",
    "TOO_LARGE", "MALFORMED", "PROTOCOL_MISMATCH", "UNKNOWN_OPINION",
)

# One table, so PRD 5.1's escalation -- suspect a typo first, send them to the
# instructor only after that -- is stated once instead of scattered through the UI.
MESSAGES: dict[str, tuple[str, str]] = {
    "UNKNOWN_ID": ("명단에서 찾을 수 없습니다.",
                   "오타가 없는지 먼저 확인해 주세요. 여러 번 실패하면 강사·조교에게 등록을 요청해 주세요."),
    "EMPTY_TEXT": ("의견을 입력해 주세요.", "내용이 있어야 지도에 올릴 수 있습니다."),
    "TEXT_TOO_LONG": ("의견이 너무 깁니다.", "20,000자 이내로 입력해 주세요. 원문은 입력창에 남아 있습니다."),
    "UNKNOWN_TARGET": ("대상 학생을 찾을 수 없습니다.", "목록에서 다시 골라 주세요."),
    "BAD_WEEK": ("주차 값이 올바르지 않습니다.", "1~4주차 중에서 골라 주세요."),
    "BAD_SOURCE": ("피드백 출처가 올바르지 않습니다.", "AI 생성 또는 사람 작성을 선택해 주세요."),
    "NOT_AUTHENTICATED": ("먼저 ID를 확인해 주세요.", "새로고침한 뒤 다시 들어와 주세요."),
    "BAD_ACCESS_CODE": ("접근 코드가 올바르지 않습니다.", ""),
    "RATE_LIMITED": ("너무 빠르게 보내고 있습니다.", "잠시 후 다시 시도해 주세요."),
    "TOO_LARGE": ("보낸 내용이 너무 큽니다.", "의견을 줄여서 다시 보내 주세요."),
    "MALFORMED": ("요청을 이해하지 못했습니다.", "새로고침한 뒤 다시 시도해 주세요."),
    "PROTOCOL_MISMATCH": ("앱이 오래된 버전입니다.", "새로고침해 주세요."),
    "UNKNOWN_OPINION": ("해당 의견을 찾을 수 없습니다.", "새로고침한 뒤 다시 시도해 주세요."),
}

MESSAGES.update({
    "NOT_AUTHORIZED": ("이 제출 내용을 수정할 권한이 없습니다.", "제출한 브라우저에서 다시 확인하거나 강사에게 요청해 주세요."),
    "UNKNOWN_SUBMISSION": ("제출 원문을 찾을 수 없습니다.", "철회되었거나 이 수업의 제출이 아닐 수 있습니다."),
    "NONCE_CONFLICT": ("이미 저장된 전송과 내용이 다릅니다.", "저장된 원문을 확인한 뒤 새 의견으로 보내 주세요."),
    "REVISION_CONFLICT": ("내용이 다른 창에서 변경되었습니다.", "최신 내용을 확인한 뒤 다시 적용해 주세요."),
    "INVALID_SEGMENTS": ("의미 단위의 범위를 확인해 주세요.", "원문을 빠짐없이 한 번씩 포함해야 합니다."),
    "INVALID_LAYOUT": ("지도 반영을 완료하지 못했습니다.", "원문은 저장되어 있습니다. 다시 시도해 주세요."),
    "QUEUE_FULL": ("처리할 의견이 많아 전송 대기 중입니다.", "입력 내용은 이 기기에 남겨 두고 잠시 후 다시 보내 주세요."),
    "STALE_CONTEXT": ("발표 대상 또는 주차가 변경되었습니다.", "현재 선택한 대상·주차를 확인한 뒤 다시 보내 주세요."),
    "CLASS_CLOSED": ("지금은 의견 제출을 받지 않습니다.", "입력 내용은 보관됩니다. 강사에게 확인해 주세요."),
    "PROCESSING_FAILED": ("원문은 저장했지만 처리를 완료하지 못했습니다.", "다시 처리하기를 눌러 주세요."),
})
ERROR_CODES = tuple(MESSAGES)

MAX_FRAME_BYTES = 256 * 1024


def error(code: str, *, fatal: bool = False) -> dict:
    if code not in ERROR_CODES:
        raise KeyError(f"unknown error code {code!r}")
    message, hint = MESSAGES[code]
    return {"t": "error", "code": code, "message": message, "hint": hint, "fatal": fatal}


def hello_ok(
    *, channel: str, rev: int, layout_rev: int, targets: list[dict],
    role: str | None = None, roster: list[dict] | None = None,
    default_source: str = "ai", default_week: int = 1, max_text_chars: int = 20000, metadata: dict | None = None,
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
    if metadata is not None:
        msg.update(metadata)
    if role is not None:
        msg["role"] = role
    if roster is not None:      # admin channel only
        msg["roster"] = roster
    return msg


def snapshot(*, rev: int, layout_rev: int, points: list[dict], projection_method=None) -> dict:
    return {"t": "snapshot", "rev": rev, "layout_rev": layout_rev, "points": points,
            **({"projection_method": projection_method} if projection_method else {})}


def delta(
    *, from_rev: int, rev: int, layout_rev: int,
    added: list[dict], moved: list[dict], drift: float = 0.0, projection_method=None,
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
        **({"projection_method": projection_method} if projection_method else {}),
        "drift": round(float(drift), 6) if math.isfinite(float(drift)) else 0.0,
    }


def ack(*, nonce: str, id: str, rev: int) -> dict:
    """Sent to the submitter alone, before the delta. Their text landed."""
    return {"t": "ack", "nonce": nonce, "id": id, "rev": rev}


def pong() -> dict:
    return {"t": "pong"}


def neighbors(*, id: str, items: list[dict], ready: bool = True) -> dict:
    """Nearest opinions to one opinion, closest first.

    Shaped as {ids, distances} to match embedding-atlas's own `data.neighbors`
    contract, so a future move to their component needs no translation.

    `ready` separates "the layout has not been computed yet" from "this opinion
    genuinely has no neighbours". Both are an empty list, and they mean opposite
    things to whoever is looking at the screen.
    """
    return {
        "t": "neighbors", "id": id, "ready": ready,
        "ids": [i["id"] for i in items],
        "distances": [round(float(i["distance"]), 6) for i in items],
    }


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
