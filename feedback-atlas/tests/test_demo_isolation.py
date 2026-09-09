"""Demo-mode regressions.

The demo page is a public practice room with public credentials and seeded
example opinions. The classroom room is the real 1-4 week corpus. These tests
pin that boundary: public demo credentials, tokens, admin codes and stored
opinions must not cross into the classroom app, and classroom secrets must not
be published by the demo metadata endpoint.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.models import Opinion  # noqa: E402
from src.protocol import PROTOCOL_VERSION  # noqa: E402
from src.store import Store  # noqa: E402
from src.textnorm import text_hash  # noqa: E402

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]

CLASSROOM_ADMIN = "classroom-admin-code"
DEMO_ADMIN = "ctp49906"

CLASSROOM_CODES = {
    "kim.seoyeon": "classroom-seoyeon",
    "kang.minsu": "classroom-minsu",
    "ta.seowoo": "classroom-seowoo",
}

CLASSROOM_ROSTER = (
    "id,display_name,role\n"
    "kim.seoyeon,김서연,student\n"
    "kang.minsu,강민수,observer\n"
    "ta.seowoo,서우,ta\n"
)

DEMO_STUDENTS = {
    "kim.seoyeon": "김서연",
    "park.junho": "박준호",
    "lee.haneul": "이하늘",
    "jung.minjae": "정민재",
    "choi.yujin": "최유진",
}
DEMO_OBSERVERS = {
    "kang.minsu": "강민수",
    "seo.jiwoo": "서지우",
    "han.doyun": "한도윤",
    "oh.sera": "오세라",
    "im.chaewon": "임채원",
}
DEMO_TAS = {
    "ta.doyo": "도요",
    "ta.seowoo": "서우",
    "ta.youngjun": "영준",
}


def _write_access_codes(path: Path, codes: dict[str, str]) -> None:
    hashes = {
        identity: hashlib.sha256(code.encode("utf-8")).hexdigest()
        for identity, code in codes.items()
    }
    path.write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")


def _seed_demo_opinions(path: Path, n: int = 30) -> None:
    store = Store(str(path))
    try:
        store.migrate()
        reviewers = [*DEMO_OBSERVERS, *DEMO_TAS]
        targets = list(DEMO_STUDENTS)
        for index in range(n):
            op = Opinion(
                id=f"demo_seed_{index + 1:02d}",
                reviewer_id=reviewers[index % len(reviewers)],
                target_id=targets[index % len(targets)],
                text=f"데모 의견 {index + 1:02d}",
                source="human" if index % 2 else "ai",
                week=(index % 4) + 1,
                timestamp=f"2026-09-08T00:{index:02d}:00.000Z",
            )
            store.insert_opinion(op, text_hash(op.text))
    finally:
        store.close()


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import create_app

    roster = tmp_path / "classroom-roster.csv"
    roster.write_text(CLASSROOM_ROSTER, encoding="utf-8")
    access = tmp_path / "classroom-access.json"
    _write_access_codes(access, CLASSROOM_CODES)
    demo_db = tmp_path / "demo.db"
    _seed_demo_opinions(demo_db)

    cfg = load_config({
        "ATLAS_ADMIN_CODE": CLASSROOM_ADMIN,
        "ATLAS_ROSTER": str(roster),
        "ATLAS_ACCESS_CODES": str(access),
        "ATLAS_DB": str(tmp_path / "classroom.db"),
        "ATLAS_ENABLE_DEMO": "1",
        "ATLAS_DEMO_DB": str(demo_db),
        "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
        "ATLAS_SKIP_WARMUP": "1",
        "ATLAS_ALLOW_INSECURE_HTTP": "1",
    })
    with TestClient(create_app(cfg)) as c:
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


def _demo_accounts(client):
    res = client.get("/demo/api/environment")
    assert res.status_code == 200
    return res.json()["accounts"]


def _account(client, *, role: str | None = None, identity: str | None = None):
    for account in _demo_accounts(client):
        if identity is not None and account["id"] == identity:
            return account
        if role is not None and account["role"] == role:
            return account
    raise AssertionError(f"no demo account matched role={role!r} identity={identity!r}")


def _wait_for_layout(client, prefix: str = "") -> None:
    endpoint = f"{prefix}/healthz" if prefix else "/healthz"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        res = client.get(endpoint)
        if res.status_code == 200 and res.json().get("layout_rev", 0) > 0:
            return
        time.sleep(0.1)
    raise AssertionError(f"{endpoint} did not report a computed layout")


def _csv_rows(response):
    assert response.status_code == 200
    return list(csv.DictReader(io.StringIO(response.content.decode("utf-8").lstrip("\ufeff"))))


def test_classroom_environment_points_to_demo_without_publishing_codes(client):
    body = client.get("/api/environment").json()
    assert body["mode"] == "classroom"
    assert body["demo_url"] == "/demo/"
    assert "accounts" not in body
    assert "admin_code" not in body
    assert CLASSROOM_ADMIN not in json.dumps(body, ensure_ascii=False)
    assert not any(code in json.dumps(body, ensure_ascii=False) for code in CLASSROOM_CODES.values())


def test_demo_environment_publishes_only_demo_credentials(client):
    body = client.get("/demo/api/environment").json()
    roles = Counter(account["role"] for account in body["accounts"])
    accounts = {(account["id"], account["display_name"], account["role"]) for account in body["accounts"]}

    assert body["mode"] == "demo"
    assert body["classroom_url"] == "/"
    assert body["admin_code"] == DEMO_ADMIN
    assert body["opinion_limit"] == 500
    assert roles == {"observer": 1}
    assert accounts == {("demo", "demo", "observer")}
    assert "auditor" not in {account["role"] for account in body["accounts"]}
    assert not any(code in json.dumps(body, ensure_ascii=False) for code in CLASSROOM_CODES.values())


def test_demo_loads_seeded_opinions_without_modifying_the_classroom_database(client):
    demo_writer = _account(client, role="observer")
    with client.websocket_connect("/demo/ws") as demo_ws:
        _hello(demo_ws, code=demo_writer["code"])
        demo_snapshot = _drain_until(demo_ws, "snapshot")

    with client.websocket_connect("/ws") as classroom_ws:
        _hello(classroom_ws, code=CLASSROOM_CODES["kang.minsu"])
        classroom_snapshot = _drain_until(classroom_ws, "snapshot")

    assert len(demo_snapshot["points"]) == 30
    assert classroom_snapshot["points"] == []
    assert client.get("/healthz").json()["opinions"] == 0


def test_demo_submission_is_visible_in_demo_and_not_in_classroom(client):
    demo_writer = _account(client, role="observer")
    demo_target = {"id": next(iter(DEMO_STUDENTS))}
    text = "데모에서만 보이는 새 의견"

    with client.websocket_connect("/demo/ws") as ws:
        _hello(ws, code=demo_writer["code"])
        _drain_until(ws, "snapshot")
        ws.send_json({
            "t": "submit",
            "nonce": "demo-only-submit",
            "target_id": demo_target["id"],
            "text": text,
            "source": "human",
            "week": 4,
        })
        _drain_until(ws, "ack")
        snapshot = _drain_until(ws, "snapshot")

    with client.websocket_connect("/ws") as classroom_ws:
        _hello(classroom_ws, code=CLASSROOM_CODES["kang.minsu"])
        classroom_snapshot = _drain_until(classroom_ws, "snapshot")

    assert any(point["text"] == text for point in snapshot["points"])
    assert all(point["text"] != text for point in classroom_snapshot["points"])
    assert client.get("/healthz").json()["opinions"] == 0


def test_demo_and_classroom_tokens_are_not_interchangeable(client):
    demo_writer = _account(client, identity="demo")

    with client.websocket_connect("/ws") as classroom_ws:
        classroom_reply = _hello(classroom_ws, code=CLASSROOM_CODES["kang.minsu"])
        classroom_token = classroom_reply["session_token"]

    with client.websocket_connect("/demo/ws") as demo_ws:
        demo_reply = _hello(demo_ws, code=demo_writer["code"])
        demo_token = demo_reply["session_token"]

    with client.websocket_connect("/demo/ws") as demo_ws:
        rejected = _hello(demo_ws, session_token=classroom_token)
        assert rejected["t"] == "error"
        assert rejected["code"] in {"BAD_ACCESS_CODE", "NOT_AUTHENTICATED"}

    with client.websocket_connect("/ws") as classroom_ws:
        rejected = _hello(classroom_ws, session_token=demo_token)
        assert rejected["t"] == "error"
        assert rejected["code"] in {"BAD_ACCESS_CODE", "NOT_AUTHENTICATED"}


def test_shared_demo_account_cannot_authenticate_to_classroom(client):
    demo_account = _account(client, identity="demo")

    with client.websocket_connect("/demo/ws") as demo_ws:
        rejected = _hello(demo_ws, code=CLASSROOM_CODES["ta.seowoo"])
        assert rejected["t"] == "error"
        assert rejected["code"] == "BAD_ACCESS_CODE"

    with client.websocket_connect("/ws") as classroom_ws:
        rejected = _hello(classroom_ws, code=demo_account["code"])
        assert rejected["t"] == "error"
        assert rejected["code"] == "BAD_ACCESS_CODE"


def test_admin_codes_unlock_only_their_own_export(client):
    demo_rows = _csv_rows(client.post("/demo/api/export.csv", json={"code": DEMO_ADMIN}))
    classroom_rows = _csv_rows(client.post("/api/export.csv", json={"code": CLASSROOM_ADMIN}))

    assert len(demo_rows) == 30
    assert classroom_rows == []
    assert client.post("/api/export.csv", json={"code": DEMO_ADMIN}).status_code == 403
    assert client.post("/demo/api/export.csv", json={"code": CLASSROOM_ADMIN}).status_code == 403


@pytest.mark.needs_duckdb
@pytest.mark.needs_pyarrow
@pytest.mark.needs_embedding_atlas
def test_demo_query_uses_the_demo_relation(client):
    writer = _account(client, role="observer")
    with client.websocket_connect("/demo/ws") as ws:
        token = _hello(ws, code=writer["code"])["session_token"]
    _wait_for_layout(client, "/demo")

    res = client.post(
        "/demo/data/query",
        json={"type": "json", "sql": "SELECT count(*) AS n FROM dataset"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    assert res.json() == [{"n": 30}]


def test_demo_redirect_keeps_demo_resources_under_demo_scope(client):
    redirected = client.get("/demo", follow_redirects=False)
    assert redirected.status_code in {307, 308}
    assert redirected.headers["location"].endswith("/demo/")

    html = client.get("/demo/").text
    assert 'src="/app.js"' not in html
    assert 'href="/app.css"' not in html


@pytest.mark.parametrize("path", ["/admin", "/demo/admin"])
def test_admin_pages_have_real_routes_and_canonical_asset_paths(client, path):
    page = client.get(path)
    assert page.status_code == 200
    assert 'id="admin-form"' in page.text
    canonical = client.get(path + "/", follow_redirects=True)
    assert canonical.status_code == 200
    assert canonical.url.path == path
