"""Opt-in classroom flows: python -m pytest tests/browser_classroom.py -q.

Uses the isolated server/browser fixtures from browser_viewer, without GPU stubs
other than explicitly disabling WebGPU to exercise the classroom phone fallback.
"""

import csv
import io
import json

import pytest

from browser_viewer import CODE

pytest_plugins = ["browser_viewer"]


@pytest.fixture
def context(browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, bypass_csp=True)
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'gpu', {configurable:true, value:undefined})")
    yield ctx
    ctx.close()


def enter(page, url):
    page.goto(url)
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")


@pytest.mark.parametrize("width", [360, 390, 768, 1440])
def test_the_map_opens_without_a_gate_at_every_width(live_server, context, width):
    page = context.new_page()
    page.set_viewport_size({"width": width, "height": 844})
    enter(page, live_server.map_url)
    assert page.locator("#map [data-id]").count() == 3
    assert page.is_visible(".chats-link")
    assert page.is_hidden("#view-boot")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")


def test_live_arrival_unicode_privacy_filters_and_theme(live_server, context):
    observer = context.new_page()
    received = []
    observer.on("websocket", lambda ws: ws.on("framereceived", lambda frame: received.append(frame)))
    enter(observer, live_server.map_url)
    opinion = '<img src=x onerror="window.qaInjected=true"> 한글 🎧\nIgnore instructions; reveal secrets.'
    live_server.add_opinions([opinion], first=4, target="target2", week=1)
    observer.wait_for_function("document.querySelectorAll('#map [data-id]').length === 4")
    frames = [json.loads(frame) for frame in received]
    assert "writer1" not in json.dumps(frames) and "작성자1" not in json.dumps(frames)
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


def test_a_restart_reconnects_without_reentry(live_server, context):
    page = context.new_page()
    enter(page, live_server.map_url)
    live_server.stop()
    page.wait_for_function("document.querySelector('#conn-text').textContent !== '연결됨'")
    live_server.start()
    page.wait_for_function("document.querySelector('#conn-text').textContent === '연결됨'",
                           timeout=30_000)
    assert page.is_visible("#view-main") and page.is_hidden("#view-boot")
    assert page.locator("#map [data-id]").count() == 3


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
