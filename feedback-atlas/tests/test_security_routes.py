"""Route-level security contract for classroom exposure.

The participant side is anonymous and read-only, so what is left to defend is the
admin side: authorship (reviewer_id) must stay behind the admin code on every
route, and nothing a participant sends can write or unlock it. Runs with a fake
embedder; no model weights or network are needed.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.config import load_config  # noqa: E402
from src.protocol import PROTOCOL_VERSION  # noqa: E402
from tests.seed_helpers import seed  # noqa: E402

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]

CODE = "open-sesame-8f3a"
ROSTER_CSV = (
    "id,display_name,role\n"
    "target1,대상1,student\n"
    "target2,대상2,student\n"
    "writer1,작성자1,observer\n"
    "writer2,작성자2,observer\n"
)
ENV = {
    "ATLAS_ADMIN_CODE": CODE,
    "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
    "ATLAS_SKIP_WARMUP": "1",
}


def _config(tmp_path, **extra):
    roster = tmp_path / "roster.csv"
    roster.write_text(ROSTER_CSV, encoding="utf-8")
    return load_config({**ENV, "ATLAS_ROSTER": str(roster),
                        "ATLAS_DB": str(tmp_path / "atlas.db"), **extra})


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import create_app

    with TestClient(create_app(_config(tmp_path, ATLAS_ALLOW_INSECURE_HTTP="1"))) as c:
        yield c


def _hello(ws, **extra):
    ws.send_json({"t": "hello", "protocol": PROTOCOL_VERSION, **extra})
    return ws.receive_json()


def _query(client, sql, *, code=None):
    body = {"type": "json", "sql": sql}
    if code is not None:
        body["code"] = code
    return client.post("/data/query", json=body)


def _seed_one(client):
    text = "누구나 조회할 수 있어야 하는 의견입니다."
    seed(client, [text])
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        res = _query(client, "SELECT text FROM dataset")
        if res.status_code == 200 and any(row["text"] == text for row in res.json()):
            return
        time.sleep(0.1)
    raise AssertionError("the viewer relation never returned the seeded opinion")


def test_credentials_on_a_participant_hello_change_nothing(client):
    """An old page may still send a code or a session token. The handshake is the
    same anonymous one either way -- the admin code included -- because admin is a
    different path, not a stronger participant."""
    for extra in ({}, {"code": "old-personal-code"}, {"session_token": "x.y"},
                  {"code": CODE}):
        with client.websocket_connect("/ws") as ws:
            reply = _hello(ws, **extra)
            assert reply["t"] == "hello_ok" and reply["channel"] == "participant"
            assert "roster" not in reply and "session_token" not in reply
            snapshot = ws.receive_json()
            assert snapshot["t"] == "snapshot"


def test_websocket_rejects_spoofed_origin(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}):
            pass


@pytest.mark.needs_duckdb
@pytest.mark.needs_pyarrow
@pytest.mark.needs_embedding_atlas
def test_data_query_answers_anyone_from_the_participant_relation(client):
    _seed_one(client)
    res = _query(client, "SELECT text FROM dataset")
    assert res.status_code == 200
    assert res.json()[0]["text"].startswith("누구나")


@pytest.mark.needs_duckdb
@pytest.mark.needs_pyarrow
@pytest.mark.needs_embedding_atlas
def test_the_anonymous_relation_has_no_reviewer_id_to_ask_for(client):
    _seed_one(client)
    res = _query(client, "SELECT reviewer_id FROM dataset")
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
def test_data_query_rejects_wrong_admin_code(client):
    res = _query(client, "SELECT * FROM dataset", code="wrong")
    assert res.status_code == 403
    assert res.json()["code"] == "BAD_ACCESS_CODE"


def test_healthz_stays_available_when_plain_http_is_disallowed(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import create_app

    with TestClient(create_app(_config(tmp_path))) as c:
        assert c.get("/healthz").status_code == 200


def test_plain_http_is_rejected_by_default(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import create_app

    with TestClient(create_app(_config(tmp_path))) as c:
        res = c.get("/")
        assert res.status_code == 403
        assert "HTTPS is required" in res.text


def test_server_main_refuses_direct_unsafe_uvicorn_launch():
    from src import server

    with pytest.raises(SystemExit) as exc:
        server.main()
    assert "scripts/class.sh start" in str(exc.value)
