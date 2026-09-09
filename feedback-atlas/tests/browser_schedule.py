"""Opt-in Playwright checks for the server-owned week UI.

Run: python -m pytest tests/browser_schedule.py -q
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

from browser_viewer import CODE, LiveServer, ROOT, _free_port, _seed_database
from tests.auth_helpers import ACCESS_CODES, write_access_codes_for

pytest_plugins = ["browser_viewer"]


class AutoWeekLiveServer(LiveServer):
    def start(self) -> None:
        env = {
            **os.environ,
            "ATLAS_ADMIN_CODE": CODE,
            "ATLAS_ROSTER": str(self.roster),
            "ATLAS_ACCESS_CODES": str(self.access),
            "ATLAS_DB": str(self.db),
            "ATLAS_ENABLE_DEMO": "1",
            "ATLAS_DEMO_DB": str(self.tmpdir / "demo.db"),
            "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
            "ATLAS_SKIP_WARMUP": "1",
            "ATLAS_ALLOW_INSECURE_HTTP": "1",
        }
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "--factory",
                "src.server:create_app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=ROOT,
            env=env,
            stdout=self.log,
            stderr=self.log,
        )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError((self.tmpdir / "server.log").read_text(encoding="utf-8"))
            with contextlib.suppress(Exception):
                urllib.request.urlopen(self.url + "healthz", timeout=1).close()
                return
            time.sleep(0.1)
        self.stop()
        raise RuntimeError("server did not start")


@pytest.fixture
def live_server():
    with tempfile.TemporaryDirectory(prefix="atlas-schedule-browser-") as tmp:
        tmpdir = Path(tmp)
        server = AutoWeekLiveServer(tmpdir)
        try:
            server.port = _free_port()
            server.url = f"http://127.0.0.1:{server.port}/"
            server.roster.write_text(
                "id,display_name,role\n"
                "target1,대상1,student\n"
                "target2,대상2,student\n"
                "writer1,작성자1,observer\n",
                encoding="utf-8",
            )
            write_access_codes_for(tmpdir, ["target1", "target2", "writer1"])
            _seed_database(server.db)
            server.start()
            yield server
        finally:
            server.close()


@pytest.fixture
def context(browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, bypass_csp=True)
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'gpu', {configurable:true, value:undefined})"
    )
    yield ctx
    ctx.close()


def enter(page, url: str) -> None:
    page.goto(url)
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])", timeout=20_000)
    page.wait_for_selector("#map [data-id], #map path[d], #map circle", state="attached", timeout=20_000)


def enter_admin(page, url: str) -> None:
    page.goto(url + "admin")
    page.wait_for_selector("#view-admin-gate:not([hidden])", timeout=20_000)
    page.fill("#admin-code", CODE)
    page.locator('#admin-form button[type="submit"]').click()
    page.wait_for_selector("#adminpanel:not([hidden])", timeout=20_000)


def test_compose_has_source_specific_placeholders_and_no_week_dropdown(live_server, context):
    page = context.new_page()
    enter(page, live_server.url)

    assert page.locator("#c-week").count() == 0
    assert page.locator("#c-hint").inner_text().strip() == ""
    assert page.locator("#c-text").get_attribute("placeholder") == "AI가 생성한 피드백을 붙여넣어 주세요."
    page.screenshot(path="/tmp/feedback-atlas-compose-ai-mobile.png", full_page=True)

    page.fill("#c-text", "작성 중인 초안")
    page.click('#c-source [data-value="human"]')
    assert page.locator("#c-text").get_attribute("placeholder") == "느낀 그대로 적어 주세요."
    assert page.input_value("#c-text") == "작성 중인 초안"
    page.screenshot(path="/tmp/feedback-atlas-compose-human-mobile.png", full_page=True)


def test_participant_sees_readonly_schedule_status(live_server, context):
    page = context.new_page()
    enter(page, live_server.url)

    assert page.locator("#c-week-status").is_visible()
    assert page.locator("#schedule-pane").is_hidden()
    assert page.locator("#schedule-open").is_hidden()


def test_admin_calendar_dialog_saves_schedule_updates(live_server, context):
    page = context.new_page()
    enter_admin(page, live_server.url)

    page.click("#schedule-open")
    page.wait_for_selector("#schedule-dialog")
    page.screenshot(path="/tmp/feedback-atlas-schedule-mobile.png", full_page=True)
    page.fill("#schedule-week-1", "2026-10-16")
    page.fill("#schedule-week-2", "2026-10-23")
    page.fill("#schedule-week-3", "2026-11-06")
    page.fill("#schedule-week-4", "2026-11-13")
    page.locator("#schedule-save").click()
    page.wait_for_function(
        "document.querySelector('#schedule-dialog').open === false"
    )

    schedule = page.request.get(live_server.url + "api/schedule").json()
    assert schedule["periods"][0]["start"] == "2026-10-16"


def test_admin_calendar_dialog_has_desktop_screenshot(live_server, browser):
    desktop = browser.new_context(viewport={"width": 1440, "height": 1000}, bypass_csp=True)
    desktop.add_init_script(
        "Object.defineProperty(navigator, 'gpu', {configurable:true, value:undefined})"
    )
    page = desktop.new_page()
    try:
        enter_admin(page, live_server.url)
        page.click("#schedule-open")
        page.wait_for_selector("#schedule-dialog")
        page.screenshot(path="/tmp/feedback-atlas-schedule-desktop.png", full_page=True)
    finally:
        desktop.close()


def test_demo_calendar_does_not_expose_real_class_schedule(live_server, context):
    page = context.new_page()
    page.goto(live_server.url + "demo/")
    page.wait_for_selector("#demo-intro:not([hidden])")
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])", timeout=20_000)

    assert page.locator("#schedule-pane").is_hidden()
    assert page.request.get(live_server.url + "demo/api/schedule").json()["demo"] is True
