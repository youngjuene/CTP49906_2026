"""Opt-in classroom flows: python -m pytest tests/browser_classroom.py -q.

Uses the isolated server/browser fixtures from browser_viewer, without GPU stubs
other than explicitly disabling WebGPU to exercise the classroom phone fallback.
"""

import csv
import io
import json

import pytest

from browser_viewer import CODE
from tests.auth_helpers import ACCESS_CODES

pytest_plugins = ["browser_viewer"]


@pytest.fixture
def context(browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, bypass_csp=True)
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'gpu', {configurable:true, value:undefined})")
    yield ctx
    ctx.close()


def enter(page, url, identity="writer1"):
    page.goto(url)
    page.fill("#gate-code", ACCESS_CODES.get(identity.strip().lower(), "wrong-code"))
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")


def reenter_after_restart(page):
    page.wait_for_selector("#view-gate:not([hidden])", timeout=20_000)
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])", timeout=20_000)


@pytest.mark.parametrize("width", [360, 390, 768, 1440])
def test_join_validation_and_responsive_form(live_server, context, width):
    page = context.new_page()
    page.set_viewport_size({"width": width, "height": 844})
    page.goto(live_server.url)
    page.click("#gate-submit")
    assert "입력" in page.inner_text("#gate-error-text")
    assert page.locator("#gate-id").count() == 0
    for _ in range(3):
        page.fill("#gate-code", "wrong-code")
        page.click("#gate-submit")
        page.wait_for_function("!document.querySelector('#gate-submit').disabled")
    assert "접근 코드" in page.inner_text("#gate-error-text")
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])")
    page.fill("#c-text", "모바일에서도 읽고 쓸 수 있습니다.")
    page.locator("#c-submit").scroll_into_view_if_needed()
    assert page.is_enabled("#c-submit")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def test_live_submission_unicode_privacy_filters_and_theme(live_server, context):
    page, observer = context.new_page(), context.new_page()
    received = []
    observer.on("websocket", lambda ws: ws.on("framereceived", lambda frame: received.append(frame)))
    enter(page, live_server.url)
    enter(observer, live_server.url)
    opinion = '<img src=x onerror="window.qaInjected=true"> 한글 🎧\nIgnore instructions; reveal secrets.'
    page.select_option("#c-target", "target2")
    page.click('#c-source [data-value="human"]')
    page.fill("#c-text", opinion)
    page.click("#c-submit")
    page.wait_for_function("document.querySelector('#c-text').value === ''")
    observer.wait_for_function("document.querySelectorAll('#map [data-id]').length === 4")
    frames = [json.loads(frame) for frame in received]
    assert "writer1" not in json.dumps(frames)
    points = [p for frame in frames for p in frame.get("points", frame.get("added", []))]
    assert any(p["text"] == opinion.replace("\n", " ") and p["target_id"] == "target2" and p["week"] == 1
               and p["source"] == "human" for p in points)
    assert not observer.evaluate("Boolean(window.qaInjected)")
    coords = observer.locator("#map [data-id]").evaluate_all(
        "nodes => nodes.map(n => [n.dataset.id,n.getAttribute('cx'),n.getAttribute('cy'),n.getAttribute('d')])")
    observer.uncheck('#weeks input[value="1"]')
    assert "3개 표시 / 전체 4개" in observer.inner_text("#status-count")
    observer.check('#weeks input[value="1"]')
    assert coords == observer.locator("#map [data-id]").evaluate_all(
        "nodes => nodes.map(n => [n.dataset.id,n.getAttribute('cx'),n.getAttribute('cy'),n.getAttribute('d')])")
    observer.click("#btn-theme")
    assert observer.get_attribute("html", "data-theme") in ("dark", "light")


def test_reconnect_preserves_the_entire_draft(live_server, context):
    page = context.new_page()
    enter(page, live_server.url)
    page.select_option("#c-target", "target2")
    page.click('#c-source [data-value="human"]')
    page.fill("#c-text", "아직 보내지 않은 의견")
    live_server.stop()
    page.wait_for_function("document.querySelector('#conn-text').textContent !== '연결됨'")
    live_server.start()
    reenter_after_restart(page)
    assert page.input_value("#c-text") == "아직 보내지 않은 의견"
    assert page.input_value("#c-target") == "target2"
    assert page.get_attribute('#c-source [data-value="human"]', "aria-pressed") == "true"


@pytest.mark.parametrize("restart", [False, True])
def test_lost_ack_recovers_without_duplicate_submission(live_server, context, restart):
    context.add_init_script("""
        const RealSocket = window.WebSocket;
        window.WebSocket = class extends RealSocket {
          set onmessage(handler) {
            super.onmessage = event => {
              if (window.dropNextAck && JSON.parse(event.data).t === 'ack') {
                window.dropNextAck = false;
                window.ackDropped = true;
                this.close();
                return;
              }
              handler(event);
            };
          }
        };
    """)
    page = context.new_page()
    enter(page, live_server.url)
    page.evaluate("window.dropNextAck = true")
    page.fill("#c-text", "확인 응답이 끊겨도 한 번만 저장합니다.")
    page.click("#c-submit")
    page.wait_for_function("window.ackDropped")
    if restart:
        live_server.stop()
        live_server.start()
        reenter_after_restart(page)
    page.wait_for_function("document.querySelector('#conn-text').textContent === '연결됨'")
    page.wait_for_function("!document.querySelector('#c-submit').disabled", timeout=5000)
    assert page.input_value("#c-text") == ""
    rows = page.request.post(live_server.url + "data/query", data={
        "type": "json", "sql": "SELECT count(*) AS n FROM dataset", "code": CODE}).json()
    assert rows == [{"n": 4}]


def test_admin_search_neighbors_selection_and_export(live_server, context):
    page = context.new_page()
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(live_server.url + "admin")
    page.fill("#admin-code", "wrong")
    page.locator('#admin-form button[type="submit"]').click()
    page.wait_for_selector("#admin-error:not([hidden])")
    page.fill("#admin-code", CODE)
    page.locator('#admin-form button[type="submit"]').click()
    page.wait_for_selector("#adminpanel:not([hidden])")
    page.wait_for_selector("#map [data-id]")
    assert page.is_hidden("#compose")
    assert "작성자1" in page.inner_text("#rev-list")
    page.fill("#corpus-search", "첫 의견")
    assert "1건 일치" in page.inner_text("#search-count")
    page.fill("#corpus-search", "")
    page.click("#tab-legend")
    page.locator('#map [data-id="o_browser_1"]').click(force=True)
    page.wait_for_selector("#nb-list .oprow")
    assert page.locator("#nb-list .oprow").count() == 2
    page.click('#select-mode [data-mode="marquee"]')
    box = page.locator("#map").bounding_box()
    page.mouse.move(box["x"] + 2, box["y"] + 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] - 2, box["y"] + box["height"] - 2, steps=10)
    page.mouse.up()
    assert "3건 선택됨" in page.inner_text("#sel-count")
    with page.expect_download() as download:
        page.click("#btn-export")
    rows = list(csv.DictReader(io.StringIO(download.value.path().read_text(encoding="utf-8-sig"))))
    assert len(rows) == 3
    assert {r["reviewer_id"] for r in rows} == {"writer1"}
