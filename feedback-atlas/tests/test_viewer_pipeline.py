"""Embedding Atlas's viewer, from a submitted sentence to a row it can query.

The viewer does not receive data. It issues SQL and expects a relation to answer,
so "the viewer works" is not a claim about the frontend -- it is a claim that text
arriving on a websocket ends up as a queryable row with coordinates and a
neighbours struct, within one recompute, without reviewer_id crossing into the
participant relation on the way.

That last clause is the reason this file exists rather than a few unit tests.
src/payloads.py enforces the privacy boundary by writing two serialisers, and a
SQL endpoint hands all of that back if it is pointed at one database: the client
writes the query, so no allowlist on the way out can hold. The boundary is
therefore a property of *which database answers*, and the only honest way to test
it is to ask the running server for reviewer_id over HTTP and watch DuckDB refuse
because the column is not there.

Runs with ATLAS_UNSAFE_FAKE_EMBEDDER=1: no weights, no network.

Run:  python -m pytest feedback-atlas/tests/test_viewer_pipeline.py
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> feedback-atlas/

from src.config import load_config  # noqa: E402
from src.protocol import PROTOCOL_VERSION  # noqa: E402
from tests.auth_helpers import ACCESS_CODES, write_access_codes  # noqa: E402

pytestmark = [
    pytest.mark.needs_fastapi, pytest.mark.needs_httpx,
    pytest.mark.needs_duckdb, pytest.mark.needs_pyarrow,
    pytest.mark.needs_embedding_atlas,
]

CODE = "open-sesame-8f3a"
ROSTER_CSV = (
    "id,display_name,role\n"
    "target1,대상1,student\n"
    "target2,대상2,student\n"
    "writer1,작성자1,observer\n"
)
# Distinctive enough that finding it in a participant response is evidence.
REVIEWER_CANARY = "writer1"


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


def _submit(client, texts):
    """Write `texts` through a real participant socket and wait for the layout."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "t": "hello", "protocol": PROTOCOL_VERSION,
            "id": "writer1", "code": ACCESS_CODES["writer1"],
        })
        assert ws.receive_json()["t"] == "hello_ok"
        assert ws.receive_json()["t"] == "snapshot"
        for i, text in enumerate(texts):
            # The submit path is rate limited to a burst of five and then one a
            # second, which is right for a person and wrong for a loop. Anything
            # past the fifth text is refused until a token refills, so a refusal
            # is waited out and retried rather than treated as a failure -- the
            # limiter is not what this file is testing.
            for _ in range(8):
                ws.send_json({"t": "submit", "nonce": f"n{i}",
                              "target_id": "target1", "text": text,
                              "source": "human", "week": 2})
                outcome = None
                for _ in range(12):
                    frame = ws.receive_json()
                    if frame.get("t") == "ack" and frame.get("nonce") == f"n{i}":
                        outcome = "ack"
                        break
                    if frame.get("t") == "error":
                        outcome = frame.get("code")
                        break
                if outcome == "ack":
                    break
                assert outcome == "RATE_LIMITED", f"submit refused: {outcome}"
                time.sleep(1.2)      # one token, plus a margin
            else:
                raise AssertionError(f"submission {i} never landed")
    # The recompute is debounced and then runs in a worker thread; the relation is
    # reloaded by commit(), before the broadcast.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        rows = client.get("/healthz").json()["viewer"]["rows"]
        if rows >= len(texts):
            return rows
        time.sleep(0.1)
    raise AssertionError("the viewer relation never caught up with the corpus")


def _participant_token(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "t": "hello", "protocol": PROTOCOL_VERSION,
            "id": "writer1", "code": ACCESS_CODES["writer1"],
        })
        hello = ws.receive_json()
        assert hello["t"] == "hello_ok"
        return hello["session_token"]


def _query(client, sql, *, code=None, kind="json"):
    body = {"type": kind, "sql": sql}
    headers = {}
    if code is not None:
        body["code"] = code
    else:
        headers["Authorization"] = f"Bearer {_participant_token(client)}"
    return client.post("/data/query", json=body, headers=headers)


def test_a_submitted_sentence_becomes_a_row_the_viewer_can_query(client):
    """The whole point, end to end: text in, queryable row out."""
    _submit(client, ["소리가 몸 안쪽에서 울렸어요.", "색이 생각보다 차가웠습니다.",
                     "질감이 손끝에 남는 느낌이었어요."])

    res = _query(client, "SELECT id, text, x, y, source, week FROM dataset ORDER BY id")
    assert res.status_code == 200, res.text
    rows = res.json()
    assert len(rows) == 3
    assert {r["source"] for r in rows} == {"human"}
    assert {r["week"] for r in rows} == {2}
    # Coordinates are real numbers, not the 0.0 a point with no layout would carry.
    assert any(r["x"] != 0.0 or r["y"] != 0.0 for r in rows)
    assert all(isinstance(r["text"], str) and r["text"] for r in rows)


def test_neighbors_column_matches_embedding_atlas_contract(client):
    """`neighbors` is a struct of two parallel arrays, readable by name.

    This is the shape its viewer reads: `{"ids": [...], "distances": [...]}` where
    ids are row ids as given by the id column. If the struct is inferred rather
    than declared, an empty first batch types it as list<null> and every later row
    fails -- so this asserts the nested access the viewer actually performs.
    """
    _submit(client, [f"의견 {i}: 서로 다른 문장입니다." for i in range(6)])

    res = _query(client, """
        SELECT id,
               len(neighbors.ids)       AS n_ids,
               len(neighbors.distances) AS n_dist,
               neighbors.ids[1]         AS nearest,
               neighbors.distances[1]   AS nearest_distance
        FROM dataset ORDER BY id
    """)
    assert res.status_code == 200, res.text
    rows = res.json()
    assert rows, "no rows came back"
    for r in rows:
        assert r["n_ids"] == r["n_dist"], "ids and distances must stay parallel"
        assert r["n_ids"] > 0, "six distinct opinions must have neighbours"
        assert r["nearest"] != r["id"], "a point must not be its own neighbour"
        assert 0.0 <= r["nearest_distance"] <= 2.0, "1 - cosine lies in [0, 2]"

    # Every id named as a neighbour is a real row, which is what makes the
    # viewer's neighbour panel able to resolve them.
    ids = {r["id"] for r in rows}
    assert {r["nearest"] for r in rows} <= ids


def test_participant_relation_cannot_be_asked_for_reviewer_id(client):
    """The boundary is the schema, not a filter.

    A rejected query is not enough on its own -- a filter could reject this one
    and miss `SELECT * ` -- so both are checked, and the wildcard is checked by
    reading the columns that come back.
    """
    _submit(client, ["작성자는 참가자 화면에 절대 나오지 않습니다."])

    direct = _query(client, "SELECT reviewer_id FROM dataset")
    assert direct.status_code == 500
    assert "reviewer_id" in direct.json()["error"]

    wildcard = _query(client, "SELECT * FROM dataset")
    assert wildcard.status_code == 200
    columns = set(wildcard.json()[0])
    assert "reviewer_id" not in columns and "reviewer_name" not in columns
    assert REVIEWER_CANARY not in wildcard.text

    # And the column really does exist for the audience entitled to it, so the
    # test above is measuring a boundary rather than a missing feature.
    admin = _query(client, "SELECT reviewer_id, reviewer_name FROM dataset", code=CODE)
    assert admin.status_code == 200
    assert admin.json()[0]["reviewer_id"] == REVIEWER_CANARY


def test_a_wrong_code_is_refused_rather_than_quietly_downgraded(client):
    """Answering from the participant relation would look right and be wrong.

    An instructor opens the viewer to see who wrote what. Silently serving the
    relation without authorship shows them a working dashboard that is missing
    the one thing they came for, with nothing on screen to say so.
    """
    _submit(client, ["코드가 틀리면 거절합니다."])
    res = _query(client, "SELECT * FROM dataset", code="not-the-code")
    assert res.status_code == 403
    assert res.json()["code"] == "BAD_ACCESS_CODE"


def test_duckdb_refuses_to_read_the_filesystem(client):
    """The endpoint runs client SQL, so this is the hardening that matters.

    Without it, `read_csv_auto('/etc/passwd')` is an ordinary query and this
    server is reachable over a public tunnel. Apple's own server sets the same two
    options; this asserts ours are actually in force, including that they cannot
    be turned back off.
    """
    _submit(client, ["파일 접근은 막혀 있어야 합니다."])

    read = _query(client, "SELECT * FROM read_csv_auto('/etc/passwd')")
    assert read.status_code == 500
    assert "error" in read.json()

    unlock = _query(client, "SET enable_external_access=true")
    assert unlock.status_code == 500


def test_arrow_is_the_wire_format_the_viewer_asks_for(client):
    """`type: "arrow"` returns Arrow IPC bytes, which is what Mosaic decodes.

    The viewer requests JSON only for small metadata reads; everything it plots
    comes back this way, so a server that only answered JSON would look healthy
    and render nothing.
    """
    import pyarrow as pa

    _submit(client, ["Arrow 형식으로 돌아와야 합니다.", "두 번째 문장."])

    res = client.post(
        "/data/query",
        json={"type": "arrow", "sql": "SELECT id, x, y FROM dataset"},
        headers={"Authorization": f"Bearer {_participant_token(client)}"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/octet-stream")

    table = pa.ipc.open_stream(pa.BufferReader(res.content)).read_all()
    assert table.num_rows == 2
    assert table.column_names == ["id", "x", "y"]


def test_healthz_reports_the_viewer_without_reporting_the_code(client):
    _submit(client, ["상태 점검."])
    body = client.get("/healthz").json()
    assert body["viewer"]["enabled"] is True
    assert body["viewer"]["rows"] == 1
    assert CODE not in client.get("/healthz").text


def test_a_roster_reload_reaches_the_viewers_relation(client):
    """SIGHUP renames somebody; the viewer's table must not keep the old name.

    The two front ends learn names differently, and that asymmetry is the whole
    defect. Websocket clients are handed the roster at handshake and re-read it on
    their next reconnect, so a rename reaches them on its own. The viewer has no
    roster: display names are columns in the relation, resolved when the relation
    was last built. Before this was wired, a name corrected mid-class stayed wrong
    in the table and in every chart grouped by it until somebody happened to
    submit an opinion and trigger a recompute.
    """
    _submit(client, ["이름이 바뀌어도 이 의견은 그대로입니다."])

    before = _query(client, "SELECT DISTINCT target_name FROM dataset").json()
    assert before[0]["target_name"] == "대상1"

    atlas = client.app.state.atlas
    roster_path = Path(atlas.cfg.roster_path)
    roster_path.write_text(
        ROSTER_CSV.replace("target1,대상1,student", "target1,고친이름,student"),
        encoding="utf-8")
    client.portal.call(atlas.reload_roster)

    after = _query(client, "SELECT DISTINCT target_name FROM dataset").json()
    assert after[0]["target_name"] == "고친이름", (
        "the relation still holds the pre-reload name")

    # And the corpus itself is untouched: a rename is a display change, not an
    # edit to anyone's opinion or to which project it is filed against.
    rows = _query(client, "SELECT target_id, text FROM dataset").json()
    assert rows[0]["target_id"] == "target1"
    assert "이름이 바뀌어도" in rows[0]["text"]


def test_a_refused_roster_reload_leaves_the_relation_alone(client):
    """A broken roster file must not blank the names already on screen."""
    _submit(client, ["잘못된 명단 파일이 화면을 지우면 안 됩니다."])
    atlas = client.app.state.atlas
    Path(atlas.cfg.roster_path).write_text("this is not,a valid roster\n", encoding="utf-8")
    client.portal.call(atlas.reload_roster)  # logs and returns; must not raise

    after = _query(client, "SELECT DISTINCT target_name FROM dataset").json()
    assert after[0]["target_name"] == "대상1"


def test_roster_reload_notifies_open_viewers_on_both_channels(client):
    """Cached charts need a notification after the relation has been refreshed."""
    _submit(client, ["명단을 고쳐도 의견과 좌표는 바뀌지 않습니다."])
    atlas = client.app.state.atlas
    before = _query(client, "SELECT id, text, x, y FROM dataset").json()
    with client.websocket_connect("/ws") as participant, \
            client.websocket_connect("/ws/admin") as admin:
        participant.send_json({"t": "hello", "protocol": PROTOCOL_VERSION,
                               "id": REVIEWER_CANARY, "code": ACCESS_CODES[REVIEWER_CANARY]})
        admin.send_json({"t": "hello", "protocol": PROTOCOL_VERSION, "code": CODE})
        for ws in (participant, admin):
            assert ws.receive_json()["t"] == "hello_ok"
            assert ws.receive_json()["t"] == "snapshot"

        Path(atlas.cfg.roster_path).write_text(
            ROSTER_CSV.replace("대상1", "고친이름"), encoding="utf-8")
        client.portal.call(atlas.reload_roster)
        for ws in (participant, admin):
            # Pings bound receive_json even when the notification is missing.
            # A pong may overtake the asynchronous broadcast, so allow that order.
            for _ in range(100):
                ws.send_json({"t": "ping"})
                frame = ws.receive_json()
                if frame == {"t": "viewer_refresh"}:
                    break
                assert frame["t"] in {"pong", "roster"}
                time.sleep(0.01)
            else:
                raise AssertionError("roster reload did not notify the open viewer")
        assert _query(client, "SELECT DISTINCT target_name FROM dataset").json() == [
            {"target_name": "고친이름"}]
        assert _query(client, "SELECT id, text, x, y FROM dataset").json() == before
