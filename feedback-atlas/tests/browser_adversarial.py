"""Opt-in interface QA scenarios using synthetic data and a loopback server.

Run: .venv/bin/python -m pytest tests/browser_adversarial.py -q
Desktop repair acceptance checks, plus the phone cases F04 (tap to read) and F05
(long tooltip stays inside the map).
"""

import csv
import io
from contextlib import suppress

import pytest

pytest.register_assert_rewrite("browser_viewer")

from browser_viewer import CODE

pytest_plugins = ["browser_viewer"]


@pytest.fixture
def context(browser):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'gpu', {configurable:true, value:undefined})"
    )
    yield ctx
    ctx.close()


@pytest.fixture
def phone_context(browser):
    """A 390x844 phone-sized window driven by a mouse (F05 hover reproduction)."""
    ctx = browser.new_context(viewport={"width": 390, "height": 844})
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'gpu', {configurable:true, value:undefined})"
    )
    yield ctx
    ctx.close()


@pytest.fixture
def touch_context(browser):
    """The same phone, but touch-only: there is no hover, only taps."""
    ctx = browser.new_context(viewport={"width": 390, "height": 844},
                              has_touch=True, is_mobile=True)
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'gpu', {configurable:true, value:undefined})"
    )
    yield ctx
    ctx.close()


LONG_OPINION = "읽을 수 있어야 하는 긴 피드백입니다. " * 40


def seed_long_opinion(page, server):
    [oid] = server.add_opinions([LONG_OPINION], first=4)
    page.wait_for_function("() => document.querySelectorAll('#map [data-id]').length === 4")
    page.click("#tab-legend")
    page.click("#btn-fit")
    return oid


def tip_bounds(page):
    return page.evaluate("""() => {
      const tip = document.querySelector('.tip'), map = document.querySelector('#map-card');
      const a = tip.getBoundingClientRect(), b = map.getBoundingClientRect();
      const text = tip.querySelector('.text');
      return {tipBottom:a.bottom, mapBottom:b.bottom, tipTop:a.top, mapTop:b.top,
              tipLeft:a.left, tipRight:a.right, mapLeft:b.left, mapRight:b.right,
              pointerEvents:getComputedStyle(tip).pointerEvents,
              textOverflowY:getComputedStyle(text).overflowY,
              textScrollable:text.scrollHeight > text.clientHeight};
    }""")


def enter(page, server):
    page.goto(server.url)
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")


def export_rows(page, server):
    response = page.request.post(server.url + "api/export.csv", data={"code": CODE})
    assert response.status == 200
    return list(csv.DictReader(io.StringIO(response.body().decode("utf-8-sig"))))


def test_slow_boot_shows_the_boot_card_until_the_app_is_ready(live_server, context):
    from playwright.sync_api import Error

    page = context.new_page()
    held = []
    page.route("**/app.js", lambda route: held.append(route))
    page.goto(live_server.url, wait_until="commit")
    try:
        page.wait_for_selector("#view-boot:not([hidden])")
        assert "불러오는 중" in page.inner_text("#boot-status")
        assert page.is_hidden("#view-main")
        for route in held:
            route.continue_()
        held.clear()
        page.unroute_all(behavior="ignoreErrors")
        page.wait_for_selector("#view-main:not([hidden])")
        assert page.is_hidden("#view-boot")
    finally:
        for route in held:
            with suppress(Error):
                route.abort()
        page.unroute_all(behavior="ignoreErrors")


def test_closed_legend_choice_survives_reconnect(live_server, context):
    page = context.new_page()
    enter(page, live_server)
    page.click("#tab-legend")
    assert page.is_hidden("#legend")
    live_server.stop()
    live_server.start()
    page.wait_for_function("() => document.querySelector('#conn-text').textContent === '연결됨'",
                           timeout=30_000)
    assert page.get_attribute("#tab-legend", "aria-pressed") == "false"
    assert page.is_hidden("#legend")


def test_mobile_tap_opens_opinion_text(live_server, touch_context):
    page = touch_context.new_page()
    enter(page, live_server)
    page.click("#tab-legend")
    page.locator('#map [data-id="o_browser_1"]').tap()
    assert page.locator(".tip .text").count() == 1
    assert "첫 의견" in page.inner_text(".tip .text")
    # A touch pointer "leaves" the instant the finger lifts; the text must outlive that.
    page.wait_for_timeout(300)
    assert page.locator(".tip .text").count() == 1


def test_tapped_opinion_can_be_dismissed(live_server, touch_context):
    page = touch_context.new_page()
    enter(page, live_server)
    page.click("#tab-legend")
    page.locator('#map [data-id="o_browser_1"]').tap()
    page.wait_for_selector(".tip .text")
    page.locator(".tip .tip-close").tap()
    assert page.locator(".tip").count() == 0


def test_long_opinion_tooltip_is_readable_within_map(live_server, phone_context):
    page = phone_context.new_page()
    enter(page, live_server)
    oid = seed_long_opinion(page, live_server)
    page.locator(f'#map [data-id="{oid}"]').hover(force=True)
    page.wait_for_selector(".tip")
    bounds = tip_bounds(page)
    assert bounds["tipBottom"] <= bounds["mapBottom"] and bounds["tipTop"] >= bounds["mapTop"], bounds
    assert bounds["tipLeft"] >= bounds["mapLeft"] and bounds["tipRight"] <= bounds["mapRight"], bounds


def test_tapped_long_opinion_is_scrollable_inside_map(live_server, touch_context):
    page = touch_context.new_page()
    enter(page, live_server)
    oid = seed_long_opinion(page, live_server)
    page.locator(f'#map [data-id="{oid}"]').tap()
    page.wait_for_selector(".tip .text")
    bounds = tip_bounds(page)
    assert bounds["tipBottom"] <= bounds["mapBottom"] and bounds["tipTop"] >= bounds["mapTop"], bounds
    # The whole opinion has to be reachable: a bounded box the reader can scroll.
    assert bounds["pointerEvents"] != "none", bounds
    assert bounds["textOverflowY"] in ("auto", "scroll") and bounds["textScrollable"], bounds
    page.evaluate("() => { const t = document.querySelector('.tip .text'); t.scrollTop = t.scrollHeight }")
    assert page.evaluate("() => { const t = document.querySelector('.tip .text');"
                         " return t.scrollTop + t.clientHeight >= t.scrollHeight - 1 }")


def test_roster_target_removal_reaches_the_legend(live_server, context):
    page = context.new_page()
    enter(page, live_server)
    assert "대상2" in page.inner_text("#legend")
    live_server.roster.write_text(
        "id,display_name,role\ntarget1,대상1,student\nwriter1,작성자1,observer\n",
        encoding="utf-8",
    )
    live_server.sighup()
    page.wait_for_function("() => !document.querySelector('#legend').textContent.includes('대상2')")
    assert "대상1" in page.inner_text("#legend")
    assert "작성자1" not in page.content()


def test_filtered_empty_status_clears_when_points_return(live_server, context):
    page = context.new_page()
    enter(page, live_server)
    page.uncheck('#weeks input[value="2"]')
    assert "0개 표시" in page.inner_text("#status-count")
    page.check('#weeks input[value="2"]')
    assert "3개 표시" in page.inner_text("#status-count")
    assert "주차가 모두 꺼져 있습니다" not in page.inner_text("#status-msg")


def test_map_zoom_updates_distance_scale(live_server, context):
    page = context.new_page()
    enter(page, live_server)
    page.click("#tab-legend")
    original = page.inner_html("#scale")
    before = page.locator("#map [data-id]").evaluate_all(
        "nodes => nodes.map(n => n.getAttribute('cx'))"
    )
    box = page.locator("#map").bounding_box()
    page.mouse.move(box["x"] + 20, box["y"] + 20)
    page.mouse.wheel(0, -500)
    page.wait_for_function(
        "before => JSON.stringify(Array.from(document.querySelectorAll('#map [data-id]'), n => n.getAttribute('cx'))) !== JSON.stringify(before)",
        arg=before,
    )
    assert page.inner_html("#scale") != original


def enter_admin(page, server):
    page.goto(server.url + "admin")
    page.fill("#admin-code", CODE)
    page.locator('#admin-form button[type="submit"]').click()
    page.wait_for_selector("#adminpanel:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")


def drag_map(page):
    box = page.locator("#map").bounding_box()
    page.mouse.move(box["x"] + 2, box["y"] + 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] - 2, box["y"] + box["height"] - 2, steps=10)
    page.mouse.up()


def test_selected_points_still_obey_week_filter(live_server, context):
    page = context.new_page()
    page.set_viewport_size({"width": 1440, "height": 1000})
    enter_admin(page, live_server)
    page.click("#tab-legend")
    page.click('#select-mode [data-mode="marquee"]')
    drag_map(page)
    assert "3건 선택됨" in page.inner_text("#sel-count")
    page.uncheck('#weeks input[value="2"]')
    assert "0개 표시" in page.inner_text("#status-count")
    opacities = page.locator("#map [data-id]").evaluate_all(
        "nodes => nodes.map(n => Number(n.getAttribute('opacity')))"
    )
    assert max(opacities) < 0.1, opacities


def test_admin_selection_mode_remains_consistent_after_reconnect(live_server, context):
    page = context.new_page()
    page.set_viewport_size({"width": 1440, "height": 1000})
    enter_admin(page, live_server)
    page.click("#tab-legend")
    page.click('#select-mode [data-mode="marquee"]')
    live_server.stop()
    live_server.start()
    page.wait_for_function("() => document.querySelector('#conn-text').textContent === '관리자 채널'")
    assert page.get_attribute('#select-mode [data-mode="marquee"]', "aria-pressed") == "true"
    drag_map(page)
    assert "3건 선택됨" in page.inner_text("#sel-count")


def test_nearest_panel_refreshes_after_new_opinion(live_server, context):
    from playwright.sync_api import expect

    admin = context.new_page()
    enter_admin(admin, live_server)
    admin.click("#tab-legend")
    admin.locator('#map [data-id="o_browser_1"]').click(force=True)
    admin.wait_for_selector("#nb-list .oprow")
    assert admin.locator("#nb-list .oprow").count() == 2
    live_server.add_opinions(["새로 도착한 가까운 의견"], first=4)
    admin.wait_for_function("() => document.querySelectorAll('#map [data-id]').length === 4")
    expect(admin.locator("#nb-list .oprow")).to_have_count(3, timeout=3000)


def test_selected_points_still_obey_search_focus(live_server, context):
    page = context.new_page()
    page.set_viewport_size({"width": 1440, "height": 1000})
    enter_admin(page, live_server)
    page.click("#tab-legend")
    page.click('#select-mode [data-mode="marquee"]')
    drag_map(page)
    assert "3건 선택됨" in page.inner_text("#sel-count")
    page.fill("#corpus-search", "첫 의견")
    assert "1건 일치" in page.inner_text("#search-count")
    opacity = float(page.get_attribute('#map [data-id="o_browser_2"]', "opacity"))
    assert opacity < 0.5, opacity


def test_viewer_backend_failure_returns_to_working_map(live_server, context):
    from playwright.sync_api import TimeoutError

    # Only the capability probe is overridden. The real viewer module, vendored
    # import, and connector run until the deliberately unavailable endpoint.
    context.add_init_script("""
        Object.defineProperty(navigator, 'gpu', {configurable:true, value:{
          wgslLanguageFeatures:new Set(),
          requestAdapter:async () => ({features:new Set(['shader-f16'])})
        }});
    """)
    page = context.new_page()
    page.route("**/data/query", lambda route: route.fulfill(
        status=503, content_type="application/json",
        body='{"error":"the viewer is not enabled on this server"}',
    ))
    enter(page, live_server)
    try:
        page.wait_for_selector("#viewer-note:not([hidden])", timeout=5000)
    except TimeoutError:
        page.click("#tab-viewer")
    page.wait_for_selector("#viewer-note:not([hidden])")
    assert "분석 보기를 불러오지 못했습니다" in page.inner_text("#viewer-note")
    assert page.is_visible("#map-card")
    assert page.get_attribute("#tab-viewer", "aria-pressed") == "false"


def test_legacy_vendored_mutations_remain_refused(live_server, context):
    # Captured from the real bundle's V3 categorical setup, independently of GPU
    # drawing. Keep this compatibility probe beside the raw trace in the report.
    # A remedy must retain the existing participant SQL privacy/write boundary.
    page = context.new_page()
    enter(page, live_server)
    control = page.request.post(live_server.url + "data/query", data={
        "type": "json", "sql": "SELECT COUNT(*) AS n FROM dataset",
    })
    assert control.status == 200
    response = page.request.post(live_server.url + "data/query", data={
        "type": "exec", "sql": '''
            ALTER TABLE dataset ADD COLUMN IF NOT EXISTS "__ev_source_id" INTEGER DEFAULT 0;
            UPDATE dataset SET "__ev_source_id" = CASE "source"::TEXT
              WHEN 'human' THEN 0 ELSE (CASE WHEN "source" IS NULL THEN 2 ELSE 1 END) END
        ''',
    })
    assert response.status == 500
    assert "exactly one statement" in response.text()
