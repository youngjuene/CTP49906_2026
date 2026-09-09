"""Desktop admin/map regression checks for the QA repair lane."""

import pytest
from playwright.sync_api import expect

pytest.register_assert_rewrite("browser_viewer")

from browser_viewer import CODE
from tests.auth_helpers import ACCESS_CODES

pytest_plugins = ["browser_viewer"]


@pytest.fixture
def context(browser):
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'gpu', {configurable:true, value:undefined})"
    )
    yield ctx
    ctx.close()


def enter(page, server, identity="writer1"):
    page.goto(server.url)
    expect(page.locator("#gate-submit")).to_be_enabled()
    page.fill("#gate-code", ACCESS_CODES[identity])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")


def enter_admin(page, server):
    page.goto(server.url + "admin")
    expect(page.locator("#admin-submit")).to_be_enabled()
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


def test_selected_points_obey_search_focus_after_selection(live_server, context):
    page = context.new_page()
    enter_admin(page, live_server)
    page.click("#tab-legend")
    page.click('#select-mode [data-mode="marquee"]')
    drag_map(page)
    assert "3건 선택됨" in page.inner_text("#sel-count")

    page.fill("#corpus-search", "첫 의견")

    assert "1건 일치" in page.inner_text("#search-count")
    opacity = float(page.get_attribute('#map [data-id="o_browser_2"]', "opacity"))
    assert opacity < 0.5, opacity


def test_admin_selection_mode_survives_reconnect(live_server, context):
    page = context.new_page()
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
    admin, participant = context.new_page(), context.new_page()
    enter_admin(admin, live_server)
    admin.click("#tab-legend")
    admin.locator('#map [data-id="o_browser_1"]').click(force=True)
    admin.wait_for_selector("#nb-list .oprow")
    assert admin.locator("#nb-list .oprow").count() == 2

    enter(participant, live_server)
    participant.fill("#c-text", "새로 도착한 가까운 의견")
    participant.click("#c-submit")
    admin.wait_for_function("() => document.querySelectorAll('#map [data-id]').length === 4")

    expect(admin.locator("#nb-list .oprow")).to_have_count(3, timeout=3000)


def test_nearest_panel_ignores_delayed_stale_nonce_response(live_server, context):
    context.add_init_script("""
        window.neighborRequests = [];
        const RealSocket = window.WebSocket;
        window.WebSocket = class extends RealSocket {
          send(data) {
            try {
              const frame = JSON.parse(data);
              if (frame.t === 'neighbors') window.neighborRequests.push(frame);
            } catch {}
            return super.send(data);
          }
        };
    """)
    page = context.new_page()
    enter_admin(page, live_server)
    page.click("#tab-legend")
    page.locator('#map [data-id="o_browser_1"]').click(force=True)
    page.wait_for_function("() => window.neighborRequests.length === 1")
    page.wait_for_selector("#nb-list .oprow")
    first_nonce = page.evaluate("() => window.neighborRequests[0].nonce")

    page.evaluate("() => document.dispatchEvent(new CustomEvent('atlas:data'))")
    page.wait_for_function("() => window.neighborRequests.length === 2")
    second_nonce = page.evaluate("() => window.neighborRequests[1].nonce")

    assert first_nonce and second_nonce and first_nonce != second_nonce
    page.evaluate(
        """nonce => document.dispatchEvent(new CustomEvent('atlas:neighbors', {
          detail: {
            id: 'o_browser_1',
            nonce,
            ready: true,
            ids: ['o_browser_2'],
            distances: [0.123],
          }
        }))""",
        first_nonce,
    )

    assert page.locator("#nb-list .oprow").count() != 1
    page.evaluate(
        """nonce => document.dispatchEvent(new CustomEvent('atlas:neighbors', {
          detail: {
            id: 'o_browser_1',
            nonce,
            ready: true,
            ids: ['o_browser_2', 'o_browser_3'],
            distances: [0.123, 0.456],
          }
        }))""",
        second_nonce,
    )
    expect(page.locator("#nb-list .oprow")).to_have_count(2)
