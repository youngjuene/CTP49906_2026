"""Malformed classroom client requests should be refused without crashing handlers."""

import csv
import io

import pytest

from test_server_routes import CODE
from tests.auth_helpers import ACCESS_CODES

pytest_plugins = ["test_server_routes"]

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]


@pytest.mark.parametrize("path", ["/api/export.csv", "/data/query"])
@pytest.mark.parametrize("body", ["null", "[]", "true", "1", '"text"'])
def test_http_endpoints_require_json_objects(client, path, body):
    response = client.post(path, content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400


@pytest.mark.parametrize("path", ["/api/export.csv", "/data/query"])
def test_unicode_wrong_admin_code_is_refused(client, path):
    response = client.post(path, json={"code": "틀린 코드 🔒", "sql": "SELECT 1"})
    assert response.status_code == 403


def test_export_selection_requires_opinion_ids(client):
    response = client.post("/api/export.csv", json={"code": CODE, "ids": [{}]})
    assert response.status_code == 400


@pytest.mark.parametrize("path", ["/api/export.csv", "/data/query"])
def test_oversized_http_requests_are_refused(client, path):
    response = client.post(path, json={"code": CODE, "sql": "SELECT 1",
                                       "padding": "x" * (256 * 1024)})
    assert response.status_code == 413


def test_unicode_wrong_websocket_code_returns_an_error(client):
    with client.websocket_connect("/ws/admin") as ws:
        ws.send_json({"t": "hello", "protocol": 1, "code": "틀린 코드 🔒"})
        assert ws.receive_json()["code"] == "BAD_ACCESS_CODE"


@pytest.mark.parametrize("text", ['=HYPERLINK("https://example.invalid", "click")',
                                 "+1+1", "-1+1", "@SUM(1,2)"])
def test_spreadsheet_export_treats_opinions_as_text(client, text):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "t": "hello", "protocol": 1,
            "id": "writer1", "code": ACCESS_CODES["writer1"],
        })
        ws.receive_json()
        ws.receive_json()
        ws.send_json({"t": "submit", "target_id": "target1", "text": text,
                      "nonce": "csv-test"})
        assert ws.receive_json()["t"] == "ack"
    response = client.post("/api/export.csv", json={"code": CODE})
    rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))
    assert rows[0]["text"] == "'" + text
    assert client.app.state.atlas.state.opinions[0].text == text
