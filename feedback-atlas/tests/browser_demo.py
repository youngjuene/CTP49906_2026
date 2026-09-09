"""Opt-in demo browser flows: python -m pytest tests/browser_demo.py -q."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

from browser_viewer import CODE, LiveServer, ROOT, _free_port
from src.models import Opinion
from src.store import Store
from src.textnorm import text_hash
from tests.auth_helpers import ACCESS_CODES, write_access_codes_for

pytest_plugins = ["browser_viewer"]

DEMO_ADMIN = "ctp49906"


def test_demo_admin_path_keeps_demo_assets_and_return_link(live_server, context):
    page = context.new_page()
    page.goto(live_server.demo_url + "admin/")
    page.wait_for_url(live_server.demo_url + "admin")
    page.wait_for_selector("#view-admin-gate:not([hidden])")
    assert page.input_value("#admin-code") == DEMO_ADMIN
    page.click("#admin-back")
    page.wait_for_url(live_server.demo_url)
    page.wait_for_selector("#demo-intro:not([hidden])")


def _write_codes(path: Path, codes: dict[str, str]) -> None:
    path.write_text(
        json.dumps({
            identity: hashlib.sha256(code.encode("utf-8")).hexdigest()
            for identity, code in codes.items()
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _seed_demo_database(path: Path) -> None:
    store = Store(str(path))
    try:
        store.migrate()
        for index, text in enumerate(("첫 데모 의견입니다.", "두 번째 데모 의견입니다.", "세 번째 데모 의견입니다."), start=1):
            op = Opinion(
                id=f"demo_browser_{index}",
                reviewer_id="kang.minsu",
                target_id="kim.seoyeon",
                text=text,
                source="human",
                week=2,
                timestamp=f"2026-09-08T00:00:0{index}.000Z",
            )
            store.insert_opinion(op, text_hash(op.text))
    finally:
        store.close()


class DemoLiveServer(LiveServer):
    def __init__(self, tmpdir: Path):
        super().__init__(tmpdir)
        self.demo_db = tmpdir / "demo.db"
        self.demo_access = tmpdir / "demo-access.json"
        self.demo_url = f"http://127.0.0.1:{self.port}/demo/"

    def start(self) -> None:
        env = {
            **os.environ,
            "ATLAS_ADMIN_CODE": CODE,
            "ATLAS_ROSTER": str(self.roster),
            "ATLAS_ACCESS_CODES": str(self.access),
            "ATLAS_DB": str(self.db),
            "ATLAS_ENABLE_DEMO": "1",
            "ATLAS_DEMO_DB": str(self.demo_db),
            "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
            "ATLAS_SKIP_WARMUP": "1",
            "ATLAS_ALLOW_INSECURE_HTTP": "1",
            "ATLAS_AUTO_WEEK": "0",
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
    with tempfile.TemporaryDirectory(prefix="atlas-demo-browser-") as tmp:
        tmpdir = Path(tmp)
        server = DemoLiveServer(tmpdir)
        try:
            server.port = _free_port()
            server.url = f"http://127.0.0.1:{server.port}/"
            server.demo_url = f"http://127.0.0.1:{server.port}/demo/"
            server.roster.write_text(
                "id,display_name,role\n"
                "target1,대상1,student\n"
                "writer1,작성자1,observer\n",
                encoding="utf-8",
            )
            write_access_codes_for(tmpdir, ["target1", "writer1"])
            _write_codes(server.demo_access, {"kang.minsu": "demo-kang-minsu"})
            _seed_demo_database(server.demo_db)
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


def _context(browser, *, width: int, height: int):
    ctx = browser.new_context(viewport={"width": width, "height": height}, bypass_csp=True)
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'gpu', {configurable:true, value:undefined})"
    )
    return ctx


def _wait_demo_gate(page) -> None:
    page.wait_for_selector("#demo-intro:not([hidden])")
    assert page.locator("#gate-form input:visible").count() == 0


def _enter_demo(page, server) -> None:
    page.goto(server.demo_url)
    _wait_demo_gate(page)
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_selector("#context-banner:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")


def test_demo_explains_practice_before_one_click_entry(live_server, context):
    page = context.new_page()

    page.goto(live_server.demo_url)
    _wait_demo_gate(page)
    page.screenshot(path="/tmp/feedback-atlas-demo-gate-mobile.png", full_page=True)

    assert page.locator("#gate-code-field").is_hidden()
    assert page.locator("#gate-code-hint").is_hidden()
    assert page.locator('#gate-form input:visible').count() == 0
    assert page.locator("#gate-submit").inner_text() == "데모 시작하기"
    assert page.locator("#gate-title").inner_text() == "feedback-atlas (demo)"
    assert page.locator("#gate-sub").is_hidden()
    assert page.locator("#gate-context-note").is_hidden()
    assert page.locator("#demo-account").count() == 0
    assert page.locator("#gate-form ol").count() == 0
    intro = page.locator("#demo-intro").bounding_box()
    button = page.locator("#gate-submit").bounding_box()
    link = page.locator("#demo-main-link").bounding_box()
    assert intro["y"] + intro["height"] <= button["y"]
    assert button["y"] + button["height"] <= link["y"]


def test_demo_submission_stays_on_demo_after_reload(live_server, context):
    page = context.new_page()

    _enter_demo(page, live_server)
    page.screenshot(path="/tmp/feedback-atlas-demo-post-login-mobile.png", full_page=True)
    page.fill("#c-text", "브라우저 데모에서만 제출되는 의견")
    page.click("#c-submit")
    page.wait_for_function("document.querySelector('#c-text').value === ''")
    page.reload()
    page.wait_for_selector("#view-main:not([hidden])")

    assert "/demo/" in page.url


def test_classroom_and_demo_sessions_are_scoped_separately(live_server, context):
    page = context.new_page()

    _enter_demo(page, live_server)

    page.goto(live_server.url)
    page.wait_for_selector("#view-gate:not([hidden])")
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])")

    page.goto(live_server.demo_url)
    page.wait_for_selector("#view-main:not([hidden])")
    assert "/demo/" in page.url


def test_desktop_demo_gate_and_post_login_are_screenshot_captured(live_server, browser):
    context = _context(browser, width=1440, height=1000)
    page = context.new_page()
    try:
        page.goto(live_server.demo_url)
        _wait_demo_gate(page)
        page.screenshot(path="/tmp/feedback-atlas-demo-gate-desktop.png", full_page=True)
        page.click("#gate-submit")
        page.wait_for_selector("#view-main:not([hidden])")
        page.wait_for_selector("#context-banner:not([hidden])")
        page.wait_for_selector("#map [data-id]", state="attached")
        page.screenshot(path="/tmp/feedback-atlas-demo-post-login-desktop.png", full_page=True)
    finally:
        context.close()
