"""Opt-in failure injection for interface/backend consistency.

Run: .venv/bin/python -m pytest tests/qa_backend_scenarios.py -q
"""

import pytest

from tests.auth_helpers import ACCESS_CODES

pytest_plugins = ["test_viewer_pipeline"]


@pytest.mark.parametrize("failure_stage", ["service", "participant", "admin"])
def test_failed_viewer_refresh_does_not_silently_serve_stale_rows(client, monkeypatch, failure_stage):
    atlas = client.app.state.atlas

    def fail_replace(*args, **kwargs):
        raise RuntimeError("QA injected relation replacement failure")

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"t": "hello", "protocol": 1, "code": ACCESS_CODES["writer1"]})
        hello = ws.receive_json()
        assert hello["t"] == "hello_ok"
        assert ws.receive_json()["t"] == "snapshot"
        target = atlas.mosaic if failure_stage == "service" else getattr(atlas.mosaic, failure_stage)
        original_replace = target.replace
        monkeypatch.setattr(target, "replace", fail_replace)
        ws.send_json({"t": "submit", "nonce": "qa-mosaic-failure", "target_id": "target1",
                      "text": "뷰어 갱신 실패 중에도 저장된 의견", "source": "human"})
        assert ws.receive_json()["t"] == "ack"
        update = ws.receive_json()
        assert update["t"] in {"snapshot", "delta"}, update
        points = update.get("points", update.get("added", []))
        assert len(points) == 1
        response = client.post("/data/query", headers={
            "authorization": "Bearer " + hello["session_token"],
        }, json={"type": "json", "sql": "SELECT COUNT(*) AS n FROM dataset"})
        health = client.get("/healthz").json()
        assert "QA injected relation replacement failure" in health["last_error"]
        # A viewer query must either report its unavailability or include the
        # accepted opinion. A successful stale count is indistinguishable from
        # a genuinely smaller corpus to the interface.
        assert response.status_code == 503, {
            "websocket_points": len(points), "query_status": response.status_code,
            "query_body": response.json(), "health": health,
        }
        assert health["viewer"]["ready"] is False
        admin_response = client.post("/data/query", json={
            "code": atlas.cfg.admin_code, "type": "json", "sql": "SELECT COUNT(*) AS n FROM dataset",
        })
        assert admin_response.status_code == 503
        monkeypatch.setattr(target, "replace", original_replace)
        assert client.portal.call(atlas.state.refresh_mosaic) is True
        recovered = client.post("/data/query", headers={
            "authorization": "Bearer " + hello["session_token"],
        }, json={"type": "json", "sql": "SELECT COUNT(*) AS n FROM dataset"})
        assert recovered.status_code == 200
        assert recovered.json() == [{"n": 1}]
        health = client.get("/healthz").json()
        assert health["viewer"]["ready"] is True
        assert health["last_error"] is None


def test_inflight_query_reports_failure_even_if_rebuild_recovers(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    atlas = client.app.state.atlas
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"t": "hello", "protocol": 1, "code": ACCESS_CODES["writer1"]})
        token = ws.receive_json()["session_token"]
        ws.receive_json()
    started, release = Event(), Event()
    query = atlas.mosaic.participant.query
    replace = atlas.mosaic.replace

    def delayed_query(sql, kind):
        result = query(sql, kind)
        started.set()
        if not release.wait(5):
            raise TimeoutError("QA did not release the query")
        return result

    def failed_replace(*args, **kwargs):
        raise RuntimeError("QA interrupted viewer refresh")

    monkeypatch.setattr(atlas.mosaic.participant, "query", delayed_query)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(client.post, "/data/query", headers={
            "authorization": "Bearer " + token,
        }, json={"type": "json", "sql": "SELECT COUNT(*) AS n FROM dataset"})
        try:
            assert started.wait(5)
            monkeypatch.setattr(atlas.mosaic, "replace", failed_replace)
            assert client.portal.call(atlas.state.refresh_mosaic) is False
            monkeypatch.setattr(atlas.mosaic, "replace", replace)
            assert client.portal.call(atlas.state.refresh_mosaic) is True
        finally:
            release.set()
        assert result.result(timeout=5).status_code == 503
    assert client.get("/healthz").json()["viewer"]["ready"] is True
