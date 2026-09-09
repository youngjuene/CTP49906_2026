"""Route-level security contract for classroom exposure.

Runs with a fake embedder and test-only access codes; no model weights or network
are needed.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.config import load_config  # noqa: E402
from src.protocol import PROTOCOL_VERSION  # noqa: E402
from tests.auth_helpers import ACCESS_CODES, write_access_codes  # noqa: E402

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]

CODE = "open-sesame-8f3a"
ROSTER_CSV = (
    "id,display_name,role\n"
    "target1,대상1,student\n"
    "target2,대상2,student\n"
    "writer1,작성자1,observer\n"
    "writer2,작성자2,observer\n"
)


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import create_app

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
        "ATLAS_AUTO_WEEK": "0",
    })
    with TestClient(create_app(cfg)) as c:
        yield c


def _hello(ws, **extra):
    ws.send_json({"t": "hello", "protocol": PROTOCOL_VERSION, **extra})
    return ws.receive_json()


def _write_codes(path: Path, codes: dict[str, str]) -> Path:
    path.write_text(
        json.dumps({
            identity: hashlib.sha256(code.encode("utf-8")).hexdigest()
            for identity, code in codes.items()
        }),
        encoding="utf-8",
    )
    return path


def _drain_until(ws, kind, limit=12):
    for _ in range(limit):
        message = ws.receive_json()
        if message["t"] == kind:
            return message
    raise AssertionError(f"no {kind!r} frame arrived")


def _participant_token(client, identity="writer1"):
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, code=ACCESS_CODES[identity])
        assert reply["t"] == "hello_ok"
        assert "session_token" in reply
        return reply["session_token"]


def _query(client, sql, *, token=None, code=None):
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = {"type": "json", "sql": sql}
    if code is not None:
        body["code"] = code
    return client.post("/data/query", json=body, headers=headers)


def _seed_one(client):
    text = "참가자 토큰으로 조회할 수 있어야 하는 의견입니다."
    with client.websocket_connect("/ws") as ws:
        _hello(ws, code=ACCESS_CODES["writer1"])
        _drain_until(ws, "snapshot")
        ws.send_json({
            "t": "submit",
            "nonce": "seed-1",
            "target_id": "target1",
            "text": text,
        })
        _drain_until(ws, "ack")
        _drain_until(ws, "delta")
    deadline = time.monotonic() + 15
    token = _participant_token(client)
    while time.monotonic() < deadline:
        res = _query(client, "SELECT text FROM dataset", token=token)
        if res.status_code == 200 and any(row["text"] == text for row in res.json()):
            return
        time.sleep(0.1)
    raise AssertionError("the viewer relation never returned the seeded opinion")


def test_participant_hello_rejects_missing_access_code(client):
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, id="writer1")
        assert reply["t"] == "error"
        assert reply["code"] == "BAD_ACCESS_CODE"
        assert reply["fatal"] is True


def test_participant_hello_rejects_wrong_access_code(client):
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, id="writer1", code="wrong-code")
        assert reply["t"] == "error"
        assert reply["code"] == "BAD_ACCESS_CODE"
        assert reply["fatal"] is True


def test_participant_hello_rejects_unknown_id_with_generic_access_error(client):
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, id="wanderer", code="wrong-code")
        assert reply["t"] == "error"
        assert reply["code"] == "BAD_ACCESS_CODE"
        assert reply["fatal"] is True


def test_participant_hello_accepts_code_only_for_student(client):
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, code=ACCESS_CODES["target1"])
        assert reply["t"] == "hello_ok"
        assert reply["channel"] == "participant"
        assert reply["role"] == "student"
        assert isinstance(reply["session_token"], str)


def test_participant_hello_accepts_code_only_for_ta(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import create_app

    roster = tmp_path / "roster.csv"
    roster.write_text(
        "id,display_name,role\n"
        "target1,대상1,student\n"
        "ta.seowoo,서우,ta\n",
        encoding="utf-8",
    )
    access = _write_codes(
        tmp_path / "access-codes.json",
        {"target1": "target-one-code", "ta.seowoo": "seowoo-fixed-code"},
    )
    cfg = load_config({
        "ATLAS_ADMIN_CODE": CODE,
        "ATLAS_ROSTER": str(roster),
        "ATLAS_ACCESS_CODES": str(access),
        "ATLAS_DB": str(tmp_path / "atlas.db"),
        "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
        "ATLAS_SKIP_WARMUP": "1",
        "ATLAS_ALLOW_INSECURE_HTTP": "1",
        "ATLAS_AUTO_WEEK": "0",
    })
    with TestClient(create_app(cfg)) as c:
        with c.websocket_connect("/ws") as ws:
            reply = _hello(ws, code="seowoo-fixed-code")
            assert reply["t"] == "hello_ok"
            assert reply["channel"] == "participant"
            assert reply["role"] == "ta"


def test_participant_hello_returns_a_session_token(client):
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, code=ACCESS_CODES["writer1"])
        assert reply["t"] == "hello_ok"
        assert reply["channel"] == "participant"
        assert reply["role"] == "observer"
        assert isinstance(reply["session_token"], str)
        assert len(reply["session_token"]) > 40


def test_participant_hello_accepts_a_session_token_without_retyping_code(client):
    token = _participant_token(client)
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, session_token=token)
        assert reply["t"] == "hello_ok"
        assert reply["channel"] == "participant"
        assert reply["role"] == "observer"


def test_participant_hello_rejects_tampered_session_token(client):
    token = _participant_token(client)
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, session_token=token + "x")
        assert reply["t"] == "error"
        assert reply["code"] == "BAD_ACCESS_CODE"
        assert reply["fatal"] is True


def test_websocket_rejects_spoofed_origin(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}):
            pass


@pytest.mark.needs_duckdb
@pytest.mark.needs_pyarrow
@pytest.mark.needs_embedding_atlas
def test_data_query_rejects_missing_participant_credentials(client):
    res = _query(client, "SELECT * FROM dataset")
    assert res.status_code == 401
    assert res.json()["code"] == "NOT_AUTHENTICATED"


@pytest.mark.needs_duckdb
@pytest.mark.needs_pyarrow
@pytest.mark.needs_embedding_atlas
def test_data_query_rejects_tampered_bearer_token(client):
    token = _participant_token(client)
    res = _query(client, "SELECT * FROM dataset", token=token + "x")
    assert res.status_code == 401
    assert res.json()["code"] == "NOT_AUTHENTICATED"


@pytest.mark.needs_duckdb
@pytest.mark.needs_pyarrow
@pytest.mark.needs_embedding_atlas
def test_data_query_allows_participant_relation_with_session_token(client):
    _seed_one(client)
    token = _participant_token(client)
    res = _query(client, "SELECT text FROM dataset", token=token)
    assert res.status_code == 200
    assert res.json()[0]["text"].startswith("참가자 토큰")


@pytest.mark.needs_duckdb
@pytest.mark.needs_pyarrow
@pytest.mark.needs_embedding_atlas
def test_participant_token_cannot_unlock_admin_relation(client):
    _seed_one(client)
    token = _participant_token(client)
    res = _query(client, "SELECT reviewer_id FROM dataset", token=token)
    assert res.status_code == 500
    assert "reviewer_id" in res.json()["error"]


@pytest.mark.needs_duckdb
@pytest.mark.needs_pyarrow
@pytest.mark.needs_embedding_atlas
def test_admin_code_still_unlocks_admin_relation(client):
    _seed_one(client)
    res = _query(client, "SELECT reviewer_id FROM dataset", code=CODE)
    assert res.status_code == 200
    assert res.json()[0]["reviewer_id"] == "writer1"


@pytest.mark.needs_duckdb
@pytest.mark.needs_pyarrow
@pytest.mark.needs_embedding_atlas
def test_participant_hello_ignores_submitted_id_and_uses_code_owner(client):
    text = "코드 소유자가 제출자로 저장되어야 합니다."
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, id="writer2", code=ACCESS_CODES["writer1"])
        assert reply["t"] == "hello_ok"
        _drain_until(ws, "snapshot")
        ws.send_json({
            "t": "submit",
            "nonce": "spoofed-id-submit",
            "target_id": "target1",
            "text": text,
        })
        _drain_until(ws, "ack")
        _drain_until(ws, "delta")

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        res = _query(client, "SELECT reviewer_id, text FROM dataset", code=CODE)
        if res.status_code == 200:
            rows = [row for row in res.json() if row["text"] == text]
            if rows:
                assert rows == [{"reviewer_id": "writer1", "text": text}]
                return
        time.sleep(0.1)
    raise AssertionError("the spoofed-id submission never reached the viewer relation")


@pytest.mark.needs_duckdb
@pytest.mark.needs_pyarrow
@pytest.mark.needs_embedding_atlas
def test_data_query_rejects_wrong_admin_code(client):
    res = _query(client, "SELECT * FROM dataset", code="wrong")
    assert res.status_code == 403
    assert res.json()["code"] == "BAD_ACCESS_CODE"


def test_submission_rate_limit_is_bound_to_identity_across_reconnects(client):
    with client.websocket_connect("/ws") as ws:
        _hello(ws, code=ACCESS_CODES["writer1"])
        _drain_until(ws, "snapshot")
        for i in range(5):
            ws.send_json({
                "t": "submit",
                "nonce": f"burst-{i}",
                "target_id": "target1",
                "text": f"연속 제출 {i}",
            })
            _drain_until(ws, "ack")

    with client.websocket_connect("/ws") as ws:
        _hello(ws, code=ACCESS_CODES["writer1"])
        _drain_until(ws, "snapshot")
        ws.send_json({
            "t": "submit",
            "nonce": "burst-after-reconnect",
            "target_id": "target1",
            "text": "재접속으로 제한을 초기화할 수 없어야 합니다.",
        })
        assert _drain_until(ws, "error")["code"] == "RATE_LIMITED"


def test_healthz_stays_available_when_plain_http_is_disallowed(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import create_app

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
    })
    with TestClient(create_app(cfg)) as c:
        assert c.get("/healthz").status_code == 200


def test_plain_http_is_rejected_by_default(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import create_app

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
    })
    with TestClient(create_app(cfg)) as c:
        res = c.get("/")
        assert res.status_code == 403
        assert "HTTPS is required" in res.text


def test_server_main_refuses_direct_unsafe_uvicorn_launch():
    from src import server

    with pytest.raises(SystemExit) as exc:
        server.main()
    assert "scripts/class.sh start" in str(exc.value)
