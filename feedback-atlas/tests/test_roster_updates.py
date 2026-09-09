"""Audience-safe live roster updates, including deployments without the viewer."""

import json

import pytest

from src.config import load_config
from tests.auth_helpers import ACCESS_CODES, write_access_codes_for

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]


@pytest.mark.parametrize("viewer_disabled", [False, True])
def test_roster_updates_reach_both_channels_without_exposing_authors(tmp_path, viewer_disabled):
    from fastapi.testclient import TestClient
    from src.server import create_app

    roster = tmp_path / "roster.csv"
    roster.write_text("id,display_name,role\ntarget1,Target One,student\ntarget2,Target Two,student\nwriter1,Private Author,observer\n")
    access = write_access_codes_for(tmp_path, ["target1", "target2", "writer1"])
    cfg = load_config({
        "ATLAS_ADMIN_CODE": "qa-roster-admin", "ATLAS_ROSTER": str(roster),
        "ATLAS_ACCESS_CODES": str(access), "ATLAS_DB": str(tmp_path / "atlas.db"),
        "ATLAS_UNSAFE_FAKE_EMBEDDER": "1", "ATLAS_SKIP_WARMUP": "1",
        "ATLAS_ALLOW_INSECURE_HTTP": "1", "ATLAS_AUTO_WEEK": "0",
        "ATLAS_DISABLE_VIEWER": "1" if viewer_disabled else "0",
    })
    with TestClient(create_app(cfg)) as client:
        with client.websocket_connect("/ws") as participant, client.websocket_connect("/ws/admin") as admin:
            participant.send_json({"t": "hello", "protocol": 1, "code": ACCESS_CODES["writer1"]})
            welcome = participant.receive_json()
            assert welcome["submission_owner"]
            assert "writer1" not in welcome["submission_owner"]
            participant.receive_json()
            admin.send_json({"t": "hello", "protocol": 1, "code": "qa-roster-admin"})
            admin.receive_json()
            admin.receive_json()
            roster.write_text("id,display_name,role\ntarget1,Renamed Target,student\ntarget3,Added Target,student\nwriter1,Renamed Author,observer\n")
            write_access_codes_for(tmp_path, ["target1", "target3", "writer1"])
            client.portal.call(client.app.state.atlas.reload_roster)
            public = participant.receive_json()
            private = admin.receive_json()
            assert public["t"] == private["t"] == "roster"
            assert [t["id"] for t in public["targets"]] == ["target1", "target3"]
            assert public["targets"][0]["display_name"] == "Renamed Target"
            assert private["targets"] == public["targets"]
            assert "roster" not in public
            assert "writer1" not in json.dumps(public)
            assert "Renamed Author" not in json.dumps(public)
            assert any(r["id"] == "writer1" and r["display_name"] == "Renamed Author"
                       for r in private["roster"])
