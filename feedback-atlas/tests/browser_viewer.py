"""Explicit Playwright regressions for the viewer lifecycle.

This file is named without the default ``test_`` prefix on purpose: run it when
browser tooling is available with ``python -m pytest tests/browser_viewer.py``.
Each test starts its own uvicorn process on a random loopback port, with a
temporary roster/database and the fake embedder.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.auth_helpers import ACCESS_CODES, write_access_codes_for  # noqa: E402

CODE = "qa-code"
ROSTER = (
    "id,display_name,role\n"
    "target1,대상1,student\n"
    "target2,대상2,student\n"
    "writer1,작성자1,observer\n"
)
STUB_VIEWER = """
window.probes = [];
window.refreshes = 0;
export const rendererAvailable = () => new Promise(resolve => window.probes.push(resolve));
export async function mount(node) { node.textContent = 'Mounted viewer'; }
export function refresh() { window.refreshes += 1; }
export function setColorScheme() {}
export function destroy() {}
"""
READY_VIEWER = """
window.refreshes = 0;
export async function rendererAvailable() { return true; }
export async function mount(node) { node.textContent = 'Mounted viewer'; }
export function refresh() { window.refreshes += 1; }
export function setColorScheme() {}
export function destroy() {}
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _seed_database(path: Path) -> None:
    from src.models import Opinion
    from src.store import Store
    from src.textnorm import text_hash

    store = Store(str(path))
    try:
        store.migrate()
        for index, text in enumerate(("첫 의견입니다.", "두 번째 의견입니다.", "세 번째 의견입니다."), start=1):
            op = Opinion(
                id=f"o_browser_{index}",
                reviewer_id="writer1",
                target_id="target1",
                text=text,
                source="human",
                week=2,
                timestamp=f"2026-09-08T00:00:0{index}.000Z",
            )
            store.insert_opinion(op, text_hash(op.text))
    finally:
        store.close()


class LiveServer:
    def __init__(self, tmpdir: Path):
        self.tmpdir = tmpdir
        self.roster = tmpdir / "roster.csv"
        self.access = tmpdir / "access-codes.json"
        self.db = tmpdir / "atlas.db"
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}/"
        self.log = (tmpdir / "server.log").open("w", encoding="utf-8")
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        env = {
            **os.environ,
            "ATLAS_ADMIN_CODE": CODE,
            "ATLAS_ROSTER": str(self.roster),
            "ATLAS_ACCESS_CODES": str(self.access),
            "ATLAS_DB": str(self.db),
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

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=15)
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=15)
        self.proc = None

    def sighup(self) -> None:
        assert self.proc is not None
        self.proc.send_signal(signal.SIGHUP)

    def close(self) -> None:
        self.stop()
        self.log.close()


@pytest.fixture
def live_server():
    with tempfile.TemporaryDirectory(prefix="atlas-browser-") as tmp:
        tmpdir = Path(tmp)
        server = LiveServer(tmpdir)
        try:
            server.roster.write_text(ROSTER, encoding="utf-8")
            write_access_codes_for(tmpdir, ["target1", "target2", "writer1"])
            _seed_database(server.db)
            server.start()
            yield server
        finally:
            server.close()


@pytest.fixture
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="chromium", args=["--no-sandbox"])
        except playwright.Error:
            browser = pw.chromium.launch(args=["--no-sandbox"])
        try:
            yield browser
        finally:
            browser.close()


def _enter(page, url: str) -> None:
    page.goto(url)
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])", timeout=20_000)
    page.wait_for_selector("#map path[d], #map circle", state="attached", timeout=20_000)


def _reenter_after_restart(page) -> None:
    page.wait_for_selector("#view-gate:not([hidden])", timeout=20_000)
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])", timeout=20_000)


def _route_viewer(page, body: str) -> None:
    page.route("**/viewer.js", lambda route: route.fulfill(content_type="text/javascript", body=body))


def test_webgpu_unavailable_keeps_the_map_visible_and_disables_the_button(live_server, browser):
    context = browser.new_context(viewport={"width": 1680, "height": 980}, bypass_csp=True)
    context.add_init_script("Object.defineProperty(navigator, 'gpu', { configurable: true, value: undefined });")
    page = context.new_page()
    try:
        _enter(page, live_server.url)
        page.wait_for_function("document.querySelector('#tab-viewer').disabled")

        assert page.get_attribute("#map-card", "hidden") is None
        assert page.get_attribute("#viewer-card", "hidden") == ""
        assert page.is_disabled("#tab-viewer")
        assert "WebGPU" in page.inner_text("#status-msg")
        assert page.eval_on_selector_all("#map path[d], #map circle", "nodes => nodes.length") > 0
    finally:
        context.close()


def test_narrow_viewport_can_open_the_viewer_manually(live_server, browser):
    context = browser.new_context(viewport={"width": 390, "height": 844}, bypass_csp=True)
    page = context.new_page()
    _route_viewer(page, READY_VIEWER)
    try:
        _enter(page, live_server.url)
        page.wait_for_function("window.refreshes === 0")

        assert page.get_attribute("#tab-viewer", "aria-pressed") == "false"
        page.click("#tab-viewer")
        page.wait_for_function("window.refreshes > 0")

        assert page.get_attribute("#tab-viewer", "aria-pressed") == "true"
        assert page.get_attribute("#viewer-card", "hidden") is None
        assert page.get_attribute("#map-card", "hidden") == ""
    finally:
        context.close()


def test_reader_closed_choice_survives_server_restart(live_server, browser):
    context = browser.new_context(viewport={"width": 1680, "height": 980}, bypass_csp=True)
    page = context.new_page()
    _route_viewer(page, READY_VIEWER)
    try:
        _enter(page, live_server.url)
        page.wait_for_function("document.querySelector('#tab-viewer').getAttribute('aria-pressed') === 'true'")

        page.click("#tab-viewer")
        page.wait_for_function("document.querySelector('#tab-viewer').getAttribute('aria-pressed') === 'false'")
        live_server.stop()
        page.wait_for_function("document.querySelector('#conn-text').textContent.trim() !== '연결됨'", timeout=20_000)
        live_server.start()
        _reenter_after_restart(page)

        assert page.get_attribute("#tab-viewer", "aria-pressed") == "false"
        assert page.get_attribute("#map-card", "hidden") is None
    finally:
        context.close()


def test_delayed_capability_probe_can_be_cancelled_while_open_is_pending(live_server, browser):
    context = browser.new_context(viewport={"width": 1680, "height": 980}, bypass_csp=True)
    page = context.new_page()
    _route_viewer(page, STUB_VIEWER)
    try:
        _enter(page, live_server.url)
        page.wait_for_function("window.probes.length === 1")

        page.click("#tab-viewer")
        page.wait_for_function("window.probes.length === 2")
        page.click("#tab-viewer")
        page.evaluate("window.probes.forEach(resolve => resolve(true))")
        page.wait_for_timeout(200)

        assert page.get_attribute("#tab-viewer", "aria-pressed") == "false"
        assert page.get_attribute("#map-card", "hidden") is None
        assert page.evaluate("window.refreshes") == 0
    finally:
        context.close()


def test_delayed_default_probe_cannot_reopen_after_an_explicit_close(live_server, browser):
    context = browser.new_context(viewport={"width": 1680, "height": 980}, bypass_csp=True)
    page = context.new_page()
    _route_viewer(page, STUB_VIEWER)
    try:
        _enter(page, live_server.url)
        page.wait_for_function("window.probes.length === 1")

        page.click("#tab-viewer")
        page.wait_for_function("window.probes.length === 2")
        page.evaluate("window.probes[1](true)")
        page.wait_for_function("document.querySelector('#tab-viewer').getAttribute('aria-pressed') === 'true'")
        page.click("#tab-viewer")
        page.evaluate("window.probes[0](true)")
        page.wait_for_timeout(200)

        assert page.get_attribute("#tab-viewer", "aria-pressed") == "false"
        assert page.get_attribute("#map-card", "hidden") is None
    finally:
        context.close()


def test_sighup_refreshes_open_viewer_and_queries_the_renamed_target(live_server, browser):
    context = browser.new_context(viewport={"width": 1680, "height": 980}, bypass_csp=True)
    page = context.new_page()
    _route_viewer(page, READY_VIEWER)
    try:
        _enter(page, live_server.url)
        page.wait_for_function("window.refreshes > 0")
        before = page.evaluate("window.refreshes")

        live_server.roster.write_text(ROSTER.replace("대상1", "고친이름"), encoding="utf-8")
        live_server.sighup()

        page.wait_for_function(f"window.refreshes > {before}", timeout=10_000)
        names = page.evaluate(
            """async () => {
                const res = await fetch('/data/query', {
                    method: 'POST',
                    headers: { 'content-type': 'application/json' },
                    body: JSON.stringify({
                        type: 'json',
                        sql: 'SELECT DISTINCT target_name FROM dataset',
                        code: %r
                    }),
                });
                return res.json();
            }""" % CODE
        )
        assert names == [{"target_name": "고친이름"}]
    finally:
        context.close()


def test_viewer_failure_displays_error_text_without_executing_markup(live_server, browser):
    context = browser.new_context(viewport={"width": 1680, "height": 980}, bypass_csp=True)
    page = context.new_page()
    _route_viewer(page, READY_VIEWER.replace(
        "node.textContent = 'Mounted viewer';",
        "throw new Error('<img src=x onerror=window.injected=true>');"))
    try:
        _enter(page, live_server.url)
        page.wait_for_selector("#viewer-note:not([hidden])")
        assert page.locator("#viewer-note img").count() == 0
        assert "<img" in page.inner_text("#viewer-note")
        assert not page.evaluate("Boolean(window.injected)")
    finally:
        context.close()
