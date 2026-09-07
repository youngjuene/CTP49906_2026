"""The two channels, end to end, over real websockets. No weights, no network.

The defect: the admin payload reached participants through a shared broadcast
helper. Every unit test still passed, because each half was correct on its own --
the leak only existed once both channels were connected to the same running
server at the same moment. So the central test here connects both and asserts, of
one broadcast, that the admin socket sees reviewer_id and the participant socket
does not.

The other thing under test is that admin is a *place*, not a permission: there is
no frame a participant socket can send to become one.

Runs with ATLAS_UNSAFE_FAKE_EMBEDDER=1, so no model is ever downloaded.

Run:  python -m pytest feedback-atlas/tests/test_server_routes.py
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.config import load_config  # noqa: E402
from src.protocol import PROTOCOL_VERSION  # noqa: E402

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]

CODE = "open-sesame-8f3a"
ROSTER_CSV = (
    "id,display_name,role\n"
    "target1,대상1,student\n"
    "target2,대상2,student\n"
    "writer1,작성자1,auditor\n"
    "writer2,작성자2,auditor\n"
)


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import create_app

    roster = tmp_path / "roster.csv"
    roster.write_text(ROSTER_CSV, encoding="utf-8")
    cfg = load_config({
        "ATLAS_ADMIN_CODE": CODE,
        "ATLAS_ROSTER": str(roster),
        "ATLAS_DB": str(tmp_path / "atlas.db"),
        "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
        "ATLAS_SKIP_WARMUP": "1",
    })
    with TestClient(create_app(cfg)) as c:
        yield c


def _hello(ws, **extra):
    ws.send_json({"t": "hello", "protocol": PROTOCOL_VERSION, **extra})
    return ws.receive_json()


def _drain_until(ws, kind, limit=8):
    for _ in range(limit):
        message = ws.receive_json()
        if message["t"] == kind:
            return message
    raise AssertionError(f"no {kind!r} frame arrived")


def test_the_rate_limiter_allows_a_burst_then_throttles_and_refills():
    """Not a security control -- the link is shared with the room by design.

    It stops one stuck client from queueing thirty recomputes and starving
    everyone else's submissions during the exact minute they are all submitting.
    """
    from src.server import RateLimiter
    limiter = RateLimiter(per_second=1.0, burst=5)
    assert [limiter.allow("c1", now=0.0) for _ in range(5)] == [True] * 5
    assert limiter.allow("c1", now=0.0) is False
    assert limiter.allow("c1", now=1.0) is True          # one token refilled
    assert limiter.allow("c2", now=0.0) is True          # buckets are per connection
    limiter.forget("c1")
    assert limiter.allow("c1", now=0.0) is True          # a reconnect starts fresh


def test_a_broken_roster_file_does_not_replace_a_working_one(tmp_path):
    """SIGHUP reload. Editing roster.csv mid-class is the supported way to add a
    late entry, and a typo saved halfway through must not lock the room out."""
    from src.config import load_config as _load
    from src.server import Atlas

    roster = tmp_path / "roster.csv"
    roster.write_text(ROSTER_CSV, encoding="utf-8")
    atlas = Atlas(_load({"ATLAS_ADMIN_CODE": CODE, "ATLAS_ROSTER": str(roster),
                         "ATLAS_DB": str(tmp_path / "a.db"),
                         "ATLAS_UNSAFE_FAKE_EMBEDDER": "1", "ATLAS_SKIP_WARMUP": "1"}))
    atlas.startup()
    assert len(atlas.roster) == 4

    roster.write_text("id,display_name,role\nbroken,X,not-a-role\n", encoding="utf-8")
    atlas.reload_roster()
    assert len(atlas.roster) == 4, "a malformed roster replaced the working one"
    assert atlas.roster.resolve("writer1") is not None

    roster.write_text(ROSTER_CSV + "writer3,작성자3,auditor\n", encoding="utf-8")
    atlas.reload_roster()
    assert len(atlas.roster) == 5
    assert atlas.state.roster.resolve("writer3") is not None, \
        "the reload did not reach the running state"
    atlas.store.close()


def test_healthz_reports_state_and_never_the_access_code(client):
    body = client.get("/healthz").json()
    assert body["ok"] is True and body["protocol"] == PROTOCOL_VERSION
    assert body["cache_key"].startswith("fake:")   # the fake embedder is visible
    assert CODE not in str(body)


def test_an_unlisted_id_never_gets_past_hello(client):
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, id="wanderer")
        assert reply["t"] == "error" and reply["code"] == "UNKNOWN_ID"
        assert reply["fatal"] is True
        assert "오타" in reply["hint"]


def test_a_listed_id_gets_the_handshake_and_a_snapshot(client):
    with client.websocket_connect("/ws") as ws:
        reply = _hello(ws, id="  WRITER1 ")      # sloppy paste still resolves
        assert reply["t"] == "hello_ok"
        assert reply["channel"] == "participant" and reply["role"] == "auditor"
        assert [t["id"] for t in reply["targets"]] == ["target1", "target2"]
        assert "roster" not in reply
        assert ws.receive_json()["t"] == "snapshot"


def test_a_stale_client_is_told_to_refresh_rather_than_mis_parsing(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"t": "hello", "protocol": PROTOCOL_VERSION + 1, "id": "writer1"})
        reply = ws.receive_json()
        assert reply["code"] == "PROTOCOL_MISMATCH" and reply["fatal"] is True


def test_the_admin_socket_refuses_a_wrong_code(client):
    with client.websocket_connect("/ws/admin") as ws:
        reply = _hello(ws, code="not-the-code")
        assert reply["t"] == "error" and reply["code"] == "BAD_ACCESS_CODE"


def test_a_participant_socket_cannot_be_upgraded_by_sending_the_code(client):
    """Admin is a different path, a different channel and a different serialiser.
    There is deliberately no frame that promotes a connection in place."""
    with client.websocket_connect("/ws") as ws:
        assert _hello(ws, id="writer1", code=CODE)["channel"] == "participant"
        _drain_until(ws, "snapshot")
        ws.send_json({"t": "hello", "protocol": PROTOCOL_VERSION, "code": CODE})
        assert ws.receive_json()["t"] == "error"


def test_a_submission_reaches_a_second_participant(client):
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        _hello(a, id="writer1")
        _drain_until(a, "snapshot")
        _hello(b, id="writer2")
        _drain_until(b, "snapshot")

        a.send_json({"t": "submit", "nonce": "n1", "target_id": "target1",
                     "text": "소리가 몸 안쪽에서 나는 것처럼 들렸어요.",
                     "source": "human", "week": 2})
        acked = _drain_until(a, "ack")
        assert acked["nonce"] == "n1" and acked["rev"] == 1

        arrived = None
        for _ in range(8):
            frame = b.receive_json()
            if frame["t"] in ("delta", "snapshot"):
                arrived = frame
                break
        assert arrived is not None, "the second client never saw the new opinion"
        points = arrived.get("added") or arrived.get("points")
        assert points[0]["text"].startswith("소리가")
        assert points[0]["week"] == 2 and points[0]["source"] == "human"


def test_one_broadcast_shows_authorship_to_admin_and_not_to_participants(client):
    """The defect this file is named for, reproduced in the only configuration
    that could ever have exposed it: both channels live at once."""
    with client.websocket_connect("/ws") as p, client.websocket_connect("/ws/admin") as a:
        _hello(p, id="writer1")
        _drain_until(p, "snapshot")
        _hello(a, code=CODE)
        _drain_until(a, "snapshot")

        p.send_json({"t": "submit", "nonce": "n1", "target_id": "target1",
                     "text": "불편했는데 그 불편함이 좋았습니다.", "source": "human"})
        _drain_until(p, "ack")

        p_frame = None
        for _ in range(8):
            f = p.receive_json()
            if f["t"] in ("delta", "snapshot"):
                p_frame = f
                break
        a_frame = None
        for _ in range(8):
            f = a.receive_json()
            if f["t"] in ("delta", "snapshot"):
                a_frame = f
                break

        p_points = (p_frame.get("added") or p_frame.get("points"))
        a_points = (a_frame.get("added") or a_frame.get("points"))
        assert "reviewer_id" not in p_points[0] and "reviewer_name" not in p_points[0]
        assert a_points[0]["reviewer_id"] == "writer1"
        assert a_points[0]["reviewer_name"] == "작성자1"
        # Same point, same coordinates: one dataset, two exposures (PRD 5.6).
        assert (p_points[0]["id"], p_points[0]["x"]) == (a_points[0]["id"], a_points[0]["x"])


def test_the_admin_handshake_carries_the_roster_and_the_participant_one_does_not(client):
    with client.websocket_connect("/ws/admin") as ws:
        reply = _hello(ws, code=CODE)
        assert reply["channel"] == "admin"
        names = {r["display_name"] for r in reply["roster"]}
        assert "작성자1" in names


def test_a_refused_submission_answers_only_the_sender(client):
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        _hello(a, id="writer1")
        _drain_until(a, "snapshot")
        _hello(b, id="writer2")
        _drain_until(b, "snapshot")
        a.send_json({"t": "submit", "nonce": "n1", "target_id": "target1", "text": "   "})
        assert _drain_until(a, "error")["code"] == "EMPTY_TEXT"
        b.send_json({"t": "ping"})
        assert b.receive_json()["t"] == "pong", "a refusal was broadcast to the room"


def test_resync_returns_a_full_snapshot_at_the_current_rev(client):
    with client.websocket_connect("/ws") as ws:
        _hello(ws, id="writer1")
        _drain_until(ws, "snapshot")
        for i in range(3):
            ws.send_json({"t": "submit", "nonce": f"n{i}", "target_id": "target1",
                          "text": f"의견 {i}"})
            _drain_until(ws, "ack")
        ws.send_json({"t": "resync", "have_rev": 0})
        snap = _drain_until(ws, "snapshot")
        assert snap["rev"] == 3 and len(snap["points"]) == 3


def test_an_oversized_or_malformed_frame_does_not_drop_the_connection(client):
    """One broken client must not be able to take its own session down, let alone
    anyone else's."""
    with client.websocket_connect("/ws") as ws:
        _hello(ws, id="writer1")
        _drain_until(ws, "snapshot")
        ws.send_text("this is not json")
        assert _drain_until(ws, "error")["code"] == "MALFORMED"
        ws.send_text('{"t":"submit","text":"' + "가" * 9000 + '"}')
        assert _drain_until(ws, "error")["code"] == "TOO_LARGE"
        ws.send_json({"t": "ping"})
        assert ws.receive_json()["t"] == "pong"


def _seed(client, n=4):
    with client.websocket_connect("/ws") as ws:
        _hello(ws, id="writer1")
        _drain_until(ws, "snapshot")
        texts = ["소리가 몸 안쪽에서 나는 것처럼 들렸어요.",
                 "소리가 몸 안쪽에서 나는 것처럼 들렸어요.",
                 "명도 대비가 낮아 세부 형태의 식별이 어렵습니다.",
                 "끝나고 나서 한참 아무 말도 못 했습니다."][:n]
        ids = []
        for i, text in enumerate(texts):
            ws.send_json({"t": "submit", "nonce": f"s{i}", "target_id": "target1",
                          "text": text, "source": "ai" if i % 2 else "human"})
            ids.append(_drain_until(ws, "ack")["id"])
        # Wait for a layout, so a neighbour query is answered from real vectors
        # rather than from a corpus that has not been projected yet.
        for _ in range(40):
            if client.get("/healthz").json()["layout_rev"] > 0:
                break
            time.sleep(0.1)
        return ids


def test_a_participant_cannot_ask_for_neighbours(client):
    """Admin-only, and refused rather than silently ignored so a client bug is
    visible. The cost of the query scales with the corpus, and the participant
    channel is the one with thirty sockets on it."""
    ids = _seed(client)
    with client.websocket_connect("/ws") as ws:
        _hello(ws, id="writer1")
        _drain_until(ws, "snapshot")
        ws.send_json({"t": "neighbors", "id": ids[0], "k": 3})
        assert _drain_until(ws, "error")["code"] == "NOT_AUTHENTICATED"


def test_the_admin_gets_neighbours_ranked_by_distance(client):
    """The workshop's own question, asked one point at a time: given what a
    person wrote, what landed nearest to it?"""
    ids = _seed(client)
    with client.websocket_connect("/ws/admin") as ws:
        _hello(ws, code=CODE)
        _drain_until(ws, "snapshot")
        ws.send_json({"t": "neighbors", "id": ids[0], "k": 3})
        frame = _drain_until(ws, "neighbors")
        assert frame["id"] == ids[0] and frame["ready"] is True
        assert ids[0] not in frame["ids"], "a point is not its own neighbour"
        assert len(frame["ids"]) == len(frame["distances"]) <= 3
        assert frame["distances"] == sorted(frame["distances"]), "not ranked by distance"
        # ids[1] is the same sentence as ids[0], so it must come first.
        assert frame["ids"][0] == ids[1]
        assert frame["distances"][0] < 1e-6


def test_an_unknown_opinion_id_is_an_error_not_an_empty_answer(client):
    """An empty list would read as 'nothing is similar', which is a different
    claim from 'that opinion does not exist'."""
    _seed(client, n=2)
    with client.websocket_connect("/ws/admin") as ws:
        _hello(ws, code=CODE)
        _drain_until(ws, "snapshot")
        ws.send_json({"t": "neighbors", "id": "o_nope"})
        assert _drain_until(ws, "error")["code"] == "UNKNOWN_OPINION"


def test_export_requires_the_access_code(client):
    """The export carries every reviewer_id in the corpus, so it is the single
    most sensitive surface in the app."""
    _seed(client, n=2)
    assert client.post("/api/export.csv", json={}).status_code == 403
    assert client.post("/api/export.csv", json={"code": "guess"}).status_code == 403
    bad = client.post("/api/export.csv", content=b"not json")
    assert bad.status_code == 400


def test_export_returns_utf8_csv_with_authorship_and_coordinates(client):
    ids = _seed(client, n=3)
    res = client.post("/api/export.csv", json={"code": CODE})
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "attachment" in res.headers["content-disposition"]
    body = res.content.decode("utf-8")
    # A BOM, so Excel opens Korean as UTF-8 rather than mojibake. The instructor
    # opening this file is the entire audience for the feature.
    assert body.startswith("\ufeff"), "no BOM; Excel will mangle the Korean"
    lines = body.lstrip("\ufeff").splitlines()
    assert lines[0].split(",")[:5] == ["id", "reviewer_id", "reviewer_name",
                                       "target_id", "target_name"]
    assert len(lines) == len(ids) + 1
    assert "writer1" in body and "작성자1" in body
    assert "소리가" in body


def test_export_can_be_narrowed_to_a_selection(client):
    """What the lasso is for: pull one cluster out of the map and take it away."""
    ids = _seed(client, n=4)
    res = client.post("/api/export.csv", json={"code": CODE, "ids": ids[:2]})
    rows = res.content.decode("utf-8").lstrip("\ufeff").splitlines()[1:]
    assert len(rows) == 2
    assert all(any(i in r for i in ids[:2]) for r in rows)


def test_submissions_survive_a_restart_of_the_app(tmp_path):
    """PRD 6, through the real startup path rather than the Store alone."""
    from fastapi.testclient import TestClient

    from src.server import create_app

    roster = tmp_path / "roster.csv"
    roster.write_text(ROSTER_CSV, encoding="utf-8")
    env = {"ATLAS_ADMIN_CODE": CODE, "ATLAS_ROSTER": str(roster),
           "ATLAS_DB": str(tmp_path / "atlas.db"),
           "ATLAS_UNSAFE_FAKE_EMBEDDER": "1", "ATLAS_SKIP_WARMUP": "1"}

    with TestClient(create_app(load_config(env))) as c:
        with c.websocket_connect("/ws") as ws:
            _hello(ws, id="writer1")
            _drain_until(ws, "snapshot")
            ws.send_json({"t": "submit", "nonce": "n1", "target_id": "target1",
                          "text": "재시작 후에도 남아야 하는 의견"})
            _drain_until(ws, "ack")

    with TestClient(create_app(load_config(env))) as c:
        assert c.get("/healthz").json()["opinions"] == 1
        with c.websocket_connect("/ws") as ws:
            _hello(ws, id="writer2")
            snap = _drain_until(ws, "snapshot")
            assert snap["points"][0]["text"] == "재시작 후에도 남아야 하는 의견"
            assert "reviewer_id" not in snap["points"][0]
