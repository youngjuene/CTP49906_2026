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
CODE = "qa-code"
ROSTER = (
    "id,display_name,role\n"
    "target1,대상1,student\n"
    "target2,대상2,student\n"
    "writer1,작성자1,auditor\n"
)
STUB_VIEWER = """
window.probes = [];
window.refreshes = 0;
export const rendererAvailable = () => new Promise(resolve => window.probes.push(resolve));
export async function mount(node) { node.textContent = 'Mounted viewer'; }
export function refresh() { window.refreshes += 1; }
export function setColorScheme() {}
"""
READY_VIEWER = """
window.refreshes = 0;
export async function rendererAvailable() { return true; }
export async function mount(node) { node.textContent = 'Mounted viewer'; }
export function refresh() { window.refreshes += 1; }
export function setColorScheme() {}
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
            "ATLAS_DB": str(self.db),
            "ATLAS_UNSAFE_FAKE_EMBEDDER": "1",
            "ATLAS_SKIP_WARMUP": "1",
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
    page.fill("#gate-id", "writer1")
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])", timeout=20_000)
    page.wait_for_selector("#map path[d], #map circle", state="attached", timeout=20_000)


def _route_viewer(page, body: str) -> None:
    page.route("**/viewer.js", lambda route: route.fulfill(content_type="text/javascript", body=body))


@pytest.mark.parametrize("width", [390, 1680])
def test_classroom_map_default_and_analysis_honestly_disabled(live_server, browser, width):
    context = browser.new_context(viewport={"width": width, "height": 980})
    page = context.new_page()
    try:
        _enter(page, live_server.url)
        assert page.is_visible('#map-card')
        assert page.is_hidden('#viewer-card')
        assert page.is_disabled('#tab-viewer')
        assert '필터·선택' in page.get_attribute('#tab-viewer', 'title')
        live_server.stop()
        live_server.start()
        page.wait_for_function("document.querySelector('#conn-text').textContent === '연결됨'")
        assert page.is_visible('#map-card')
    finally:
        context.close()


def test_sighup_updates_roster_and_mosaic_without_enabling_analysis(live_server, browser):
    context = browser.new_context(viewport={"width": 1680, "height": 980})
    page = context.new_page()
    try:
        _enter(page, live_server.url)
        live_server.roster.write_text(ROSTER.replace('대상1', '고친이름'), encoding='utf-8')
        live_server.sighup()
        page.wait_for_function("document.querySelector('#c-target').textContent.includes('고친이름')")
        names = page.request.post(live_server.url+'data/query',data={
            'type':'json', 'sql':'SELECT DISTINCT target_name FROM dataset'}).json()
        assert names == [{'target_name':'고친이름'}]
        assert page.is_disabled('#tab-viewer')
    finally:
        context.close()
