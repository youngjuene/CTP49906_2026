"""Server-clock class schedule regressions.

Auto-week mode is the classroom default: participants do not choose a week, and
the server files submissions by its own Asia/Seoul clock. These tests keep the
schedule contract at the websocket and REST boundary, where stale browsers and
manual API calls can otherwise smuggle a client week back in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.protocol import PROTOCOL_VERSION  # noqa: E402
from tests.auth_helpers import ACCESS_CODES, write_access_codes  # noqa: E402

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]

CODE = "schedule-admin-code"
ROSTER_CSV = (
    "id,display_name,role\n"
    "target1,대상1,student\n"
    "target2,대상2,student\n"
    "writer1,작성자1,observer\n"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.server as server
    from src.server import create_app

    clock = {"now": "2026-10-15T00:00:00.000Z"}
    monkeypatch.setattr(server, "_now_iso", lambda: clock["now"])
    roster = tmp_path / "roster.csv"
    roster.write_text(ROSTER_CSV, encoding="utf-8")
    access = write_access_codes(tmp_path)
    cfg = load_config({
        "ATLAS_ADMIN_CODE": CODE,
        "ATLAS_ROSTER": str(roster),
        "ATLAS_ACCESS_CODES": str(access),
        "ATLAS_DB": str(tmp_path / "atlas.db"),
        "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
        "ATLAS_SKIP_WARMUP": "1",
        "ATLAS_ALLOW_INSECURE_HTTP": "1",
    })
    with TestClient(create_app(cfg)) as c:
        c.clock = clock
        yield c


def _hello(ws, **extra):
    ws.send_json({"t": "hello", "protocol": PROTOCOL_VERSION, **extra})
    return ws.receive_json()


def _drain_until(ws, kind: str, limit: int = 12):
    for _ in range(limit):
        frame = ws.receive_json()
        if frame["t"] == kind:
            return frame
    raise AssertionError(f"no {kind!r} frame arrived")


def _submit(client, *, nonce: str = "schedule-submit", **body):
    payload = {
        "t": "submit",
        "nonce": nonce,
        "target_id": "target1",
        "text": "서버 시간이 주차를 정해야 합니다.",
        "source": "human",
        **body,
    }
    with client.websocket_connect("/ws") as ws:
        _hello(ws, code=ACCESS_CODES["writer1"])
        _drain_until(ws, "snapshot")
        ws.send_json(payload)
        return ws.receive_json()


def _admin_schedule(client, **body):
    return client.put("/api/schedule", json={"code": CODE, **body})


def test_public_schedule_reports_the_default_class_periods(client):
    body = client.get("/api/schedule").json()

    assert body["timezone"] == "Asia/Seoul"
    assert body["automatic"] is True
    assert body["accepting"] is True
    assert body["current_week"] == 1
    assert body["periods"] == [
        {"week": 1, "start": "2026-10-15", "end": "2026-10-21"},
        {"week": 2, "start": "2026-10-22", "end": "2026-10-28"},
        {"week": 3, "start": "2026-11-05", "end": "2026-11-11"},
        {"week": 4, "start": "2026-11-12", "end": "2026-11-18"},
    ]


def test_hello_ok_includes_schedule_metadata(client):
    with client.websocket_connect("/ws") as ws:
        hello = _hello(ws, code=ACCESS_CODES["writer1"])

    assert hello["schedule"]["current_week"] == 1
    assert hello["schedule"]["accepting"] is True
    assert hello["schedule"]["demo"] is False
    assert hello["schedule"]["automatic"] is True


def test_submission_week_comes_from_server_clock_not_client_payload(client):
    client.clock["now"] = "2026-10-22T02:00:00.000Z"
    _submit(client, week=4, timestamp="2099-01-01T00:00:00.000Z")

    with client.websocket_connect("/ws/admin") as ws:
        _hello(ws, code=CODE)
        snapshot = _drain_until(ws, "snapshot")

    assert snapshot["points"][0]["week"] == 2
    assert snapshot["points"][0]["timestamp"] == "2026-10-22T02:00:00.000Z"


def test_submission_during_midterm_break_is_refused(client):
    client.clock["now"] = "2026-10-29T02:00:00.000Z"
    reply = _submit(client, week=2)

    assert reply["t"] == "error"
    assert reply["code"] == "OUTSIDE_CLASS_PERIOD"


def test_replayed_nonce_gets_ack_even_after_class_period_closes(client):
    first = _submit(client, nonce="retry-after-close")
    client.clock["now"] = "2026-11-19T02:00:00.000Z"
    replay = _submit(client, nonce="retry-after-close", text="재전송 본문은 저장되지 않습니다.")

    assert first["t"] == "ack"
    assert replay["t"] == "ack"
    assert replay["id"] == first["id"]


def test_admin_can_replace_the_schedule_with_matching_revision(client):
    current = client.get("/api/schedule").json()
    response = _admin_schedule(
        client,
        expected_revision=current["revision"],
        starts=["2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13"],
    )

    assert response.status_code == 200
    assert response.json()["revision"] == current["revision"] + 1
    assert response.json()["periods"][0] == {"week": 1, "start": "2026-10-16", "end": "2026-10-22"}


def test_admin_schedule_update_requires_the_admin_code(client):
    current = client.get("/api/schedule").json()
    response = client.put("/api/schedule", json={
        "code": "wrong",
        "expected_revision": current["revision"],
        "starts": ["2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13"],
    })

    assert response.status_code == 403
    assert response.json()["code"] == "BAD_ACCESS_CODE"


def test_admin_schedule_update_rejects_stale_revision(client):
    current = client.get("/api/schedule").json()
    assert _admin_schedule(
        client,
        expected_revision=current["revision"],
        starts=["2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13"],
    ).status_code == 200

    stale = _admin_schedule(
        client,
        expected_revision=current["revision"],
        starts=["2026-10-17", "2026-10-24", "2026-11-07", "2026-11-14"],
    )

    assert stale.status_code == 409
    assert stale.json()["code"] == "STALE_SCHEDULE"


def test_admin_schedule_update_rejects_overlapping_periods(client):
    current = client.get("/api/schedule").json()
    response = _admin_schedule(
        client,
        expected_revision=current["revision"],
        starts=["2026-10-15", "2026-10-20", "2026-11-05", "2026-11-12"],
    )

    assert response.status_code == 400
    assert response.json()["code"] == "BAD_SCHEDULE"


def test_schedule_update_persists_without_rewriting_existing_opinions(client):
    _submit(client)
    current = client.get("/api/schedule").json()
    assert _admin_schedule(
        client,
        expected_revision=current["revision"],
        starts=["2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13"],
    ).status_code == 200

    with client.websocket_connect("/ws/admin") as ws:
        _hello(ws, code=CODE)
        snapshot = _drain_until(ws, "snapshot")

    assert snapshot["points"][0]["week"] == 1
