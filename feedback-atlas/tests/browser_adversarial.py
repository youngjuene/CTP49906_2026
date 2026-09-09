"""Opt-in interface QA scenarios using synthetic data and a loopback server.

Run: .venv/bin/python -m pytest tests/browser_adversarial.py -q
Desktop repair acceptance checks. Mobile-only cases remain explicit exclusions.
"""

import csv
import io
from contextlib import suppress

import pytest

pytest.register_assert_rewrite("browser_viewer")

from browser_viewer import CODE
from tests.auth_helpers import ACCESS_CODES

pytest_plugins = ["browser_viewer"]


@pytest.fixture
def context(browser):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script(
        "Object.defineProperty(navigator, 'gpu', {configurable:true, value:undefined})"
    )
    yield ctx
    ctx.close()


def enter(page, server, identity="writer1"):
    page.goto(server.url)
    page.fill("#gate-code", ACCESS_CODES[identity])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")


def export_rows(page, server):
    response = page.request.post(server.url + "api/export.csv", data={"code": CODE})
    assert response.status == 200
    return list(csv.DictReader(io.StringIO(response.body().decode("utf-8-sig"))))


@pytest.mark.parametrize("delayed_path", ["api/environment", "app.js"])
def test_slow_environment_boot_does_not_put_access_code_in_url(live_server, context, delayed_path):
    from playwright.sync_api import Error
    from playwright.sync_api import expect

    page = context.new_page()
    held = []
    page.route("**/" + delayed_path, lambda route: held.append(route))
    page.goto(live_server.url, wait_until="commit")
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    try:
        assert page.is_disabled("#gate-submit")
        page.press("#gate-code", "Enter")
        page.wait_for_timeout(100)
        assert "code=" not in page.url, page.url
        assert ACCESS_CODES["writer1"] not in page.url
        for route in held:
            route.continue_()
        held.clear()
        page.unroute_all(behavior="ignoreErrors")
        expect(page.locator("#gate-submit")).to_be_enabled()
        page.fill("#gate-code", ACCESS_CODES["writer1"])
        page.click("#gate-submit")
        page.wait_for_selector("#view-main:not([hidden])")
    finally:
        for route in held:
            with suppress(Error):
                route.abort()
        page.unroute_all(behavior="ignoreErrors")


def test_pending_receipt_cannot_be_replayed_as_a_different_person(live_server, context):
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
    enter(page, live_server)
    page.evaluate("window.dropNextAck = true")
    draft = "한 사람에게만 귀속되어야 하는 보류 중인 의견"
    page.fill("#c-text", draft)
    page.click("#c-submit")
    page.wait_for_function("() => window.ackDropped")
    live_server.stop()
    live_server.start()
    page.wait_for_selector("#view-gate:not([hidden])", timeout=20_000)
    page.fill("#gate-code", ACCESS_CODES["target2"])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_function("() => !document.querySelector('#c-submit').disabled")
    rows = [row for row in export_rows(page, live_server) if row["text"] == draft]
    assert len(rows) == 1 and rows[0]["reviewer_id"] == "writer1", rows
    assert page.input_value("#c-text") == draft


def test_closed_compose_choice_survives_reconnect(live_server, context):
    page = context.new_page()
    enter(page, live_server)
    page.click("#tab-compose")
    assert page.is_hidden("#compose")
    live_server.stop()
    live_server.start()
    page.wait_for_selector("#view-gate:not([hidden])", timeout=20_000)
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])")
    assert page.get_attribute("#tab-compose", "aria-pressed") == "false"
    assert page.is_hidden("#compose")


@pytest.mark.skip(reason="F04: mobile-only behavior excluded by user")
def test_mobile_tap_opens_opinion_text(live_server, context):
    page = context.new_page()
    enter(page, live_server)
    page.click("#tab-legend")
    point = page.locator('#map [data-id="o_browser_1"]')
    point.tap()
    assert page.locator(".tip .text").count() == 1
    assert "첫 의견" in page.inner_text(".tip .text")


@pytest.mark.skip(reason="F05: phone-only reproduction; desktop 1440x900 replay passed")
def test_long_opinion_tooltip_is_readable_within_map(live_server, context):
    page = context.new_page()
    enter(page, live_server)
    opinion = "읽을 수 있어야 하는 긴 피드백입니다. " * 40
    page.fill("#c-text", opinion)
    page.click("#c-submit")
    page.wait_for_function("() => document.querySelector('#c-text').value === ''")
    page.wait_for_function("() => document.querySelectorAll('#map [data-id]').length === 4")
    page.click("#tab-legend")
    page.click("#btn-fit")
    rows = export_rows(page, live_server)
    oid = next(row["id"] for row in rows if row["text"] == opinion.strip())
    page.locator(f'#map [data-id="{oid}"]').hover(force=True)
    page.wait_for_selector(".tip")
    bounds = page.evaluate("""() => {
      const tip = document.querySelector('.tip'), map = document.querySelector('#map-card');
      const a = tip.getBoundingClientRect(), b = map.getBoundingClientRect();
      return {tipBottom:a.bottom, mapBottom:b.bottom, tipTop:a.top, mapTop:b.top,
              tipHeight:a.height, mapHeight:b.height,
              pointerEvents:getComputedStyle(tip).pointerEvents,
              overflow:getComputedStyle(map).overflow};
    }""")
    page.screenshot(path="/tmp/feedback-atlas-qa-long-tooltip.png", full_page=True)
    assert bounds["tipBottom"] <= bounds["mapBottom"] and bounds["tipTop"] >= bounds["mapTop"], bounds


def test_submission_validation_and_rate_limit_preserve_draft(live_server, context):
    page = context.new_page()
    enter(page, live_server)
    page.fill("#c-text", "   ")
    page.click("#c-submit")
    assert "입력" in page.inner_text("#c-hint")
    assert page.is_enabled("#c-submit")
    # The server intentionally allows a five-message burst. Exercise an actual
    # refusal by submitting sequentially within that burst's refill interval.
    result = page.evaluate("""async () => {
      for (let i = 0; i < 12; i++) {
        const draft = '빠른 제출에서도 보존할 초안 ' + i;
        document.querySelector('#c-text').value = draft;
        const response = new Promise(resolve => {
          const done = event => {
            document.removeEventListener('atlas:ack', done);
            document.removeEventListener('atlas:error', done);
            resolve({type:event.type, detail:event.detail});
          };
          document.addEventListener('atlas:ack', done);
          document.addEventListener('atlas:error', done);
        });
        document.querySelector('#compose-form').requestSubmit();
        const event = await response;
        if (event.type === 'atlas:error') return {draft, ...event};
      }
      return null;
    }""")
    assert result and result["detail"]["code"] == "RATE_LIMITED", result
    assert page.input_value("#c-text") == result["draft"]
    assert page.is_enabled("#c-submit")
    assert "잠시" in page.inner_text("#c-hint")


def test_editing_text_during_ack_does_not_erase_new_draft(live_server, context):
    context.add_init_script("""
        const RealSocket = window.WebSocket;
        window.WebSocket = class extends RealSocket {
          set onmessage(handler) {
            super.onmessage = event => {
              if (JSON.parse(event.data).t === 'ack') {
                window.releaseAck = () => handler(event);
                return;
              }
              handler(event);
            };
          }
        };
    """)
    page = context.new_page()
    enter(page, live_server)
    page.fill("#c-text", "먼저 제출한 의견")
    page.click("#c-submit")
    page.wait_for_function("() => Boolean(window.releaseAck)")
    page.fill("#c-text", "다음 의견 초안")
    page.evaluate("window.releaseAck()")
    assert page.input_value("#c-text") == "다음 의견 초안"
    assert page.is_enabled("#c-submit")


def test_roster_target_removal_requires_explicit_reselection(live_server, context):
    context.add_init_script("""
        const RealSocket = window.WebSocket;
        window.WebSocket = class extends RealSocket {
          set onmessage(handler) {
            super.onmessage = event => {
              if (JSON.parse(event.data).t === 'viewer_refresh') window.rosterRefreshed = true;
              handler(event);
            };
          }
        };
    """)
    page = context.new_page()
    enter(page, live_server)
    page.select_option("#c-target", "target2")
    page.fill("#c-text", "대상이 사라져도 남아야 하는 초안")
    live_server.roster.write_text(
        "id,display_name,role\ntarget1,대상1,student\nwriter1,작성자1,observer\n",
        encoding="utf-8",
    )
    live_server.sighup()
    page.wait_for_function("() => window.rosterRefreshed")
    assert page.input_value("#c-target") == "", page.inner_html("#c-target")
    assert "다시 선택" in page.inner_text("#c-hint")
    assert page.input_value("#c-text") == "대상이 사라져도 남아야 하는 초안"
    page.select_option("#c-target", "target1")
    page.click("#c-submit")
    page.wait_for_function("() => document.querySelector('#c-text').value === ''")


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


def set_dates(page, starts):
    for week, start in enumerate(starts, start=1):
        page.fill(f"#schedule-week-{week}", start)


def test_schedule_concurrent_edits_require_reload_without_losing_draft(live_server, context):
    first, second = context.new_page(), context.new_page()
    for page in (first, second):
        enter_admin(page, live_server)
        page.click("#schedule-open")
    first_starts = ["2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13"]
    second_starts = ["2026-10-17", "2026-10-24", "2026-11-07", "2026-11-14"]
    set_dates(first, first_starts)
    set_dates(second, second_starts)
    first.click("#schedule-save")
    first.wait_for_function("() => !document.querySelector('#schedule-dialog').open")
    second.wait_for_selector("#schedule-error:not([hidden])")
    assert second.input_value("#schedule-week-1") == second_starts[0]
    with second.expect_response(lambda response: response.request.method == "PUT") as response:
        second.click("#schedule-save")
    assert response.value.status == 409
    assert second.input_value("#schedule-week-1") == second_starts[0]
    second.click("#schedule-reload")
    second.wait_for_function("() => document.querySelector('#schedule-week-1').value === '2026-10-16'")
    set_dates(second, second_starts)
    second.click("#schedule-save")
    second.wait_for_function("() => !document.querySelector('#schedule-dialog').open")
    saved = second.request.get(live_server.url + "api/schedule").json()
    assert [p["start"] for p in saved["periods"]] == second_starts


def test_schedule_overlap_validation_and_cancel_preserve_server_dates(live_server, context):
    page = context.new_page()
    enter_admin(page, live_server)
    original = page.request.get(live_server.url + "api/schedule").json()
    page.click("#schedule-open")
    page.fill("#schedule-week-2", "2026-10-16")
    page.click("#schedule-save")
    assert page.is_visible("#schedule-error")
    assert "겹치지" in page.inner_text("#schedule-error")
    assert page.is_enabled("#schedule-save")
    page.click("#schedule-cancel")
    assert page.request.get(live_server.url + "api/schedule").json()["revision"] == original["revision"]
    page.click("#schedule-open")
    assert page.input_value("#schedule-week-2") == original["periods"][1]["start"]


def test_schedule_failed_save_retains_dates_and_can_retry(live_server, context):
    page = context.new_page()
    enter_admin(page, live_server)
    page.click("#schedule-open")
    starts = ["2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13"]
    set_dates(page, starts)
    page.route("**/api/schedule", lambda route: route.abort())
    page.click("#schedule-save")
    page.wait_for_selector("#schedule-error:not([hidden])")
    assert "네트워크" in page.inner_text("#schedule-error")
    assert page.is_enabled("#schedule-save")
    assert page.input_value("#schedule-week-1") == starts[0]
    page.unroute("**/api/schedule")
    page.click("#schedule-save")
    page.wait_for_function("() => !document.querySelector('#schedule-dialog').open")
    saved = page.request.get(live_server.url + "api/schedule").json()
    assert [p["start"] for p in saved["periods"]] == starts


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
    token = page.evaluate("sessionStorage.getItem('atlas.session_token:classroom')")
    headers = {"authorization": "Bearer " + token}
    control = page.request.post(live_server.url + "data/query", headers=headers, data={
        "type": "json", "sql": "SELECT COUNT(*) AS n FROM dataset",
    })
    assert control.status == 200
    response = page.request.post(live_server.url + "data/query", headers=headers, data={
        "type": "exec", "sql": '''
            ALTER TABLE dataset ADD COLUMN IF NOT EXISTS "__ev_source_id" INTEGER DEFAULT 0;
            UPDATE dataset SET "__ev_source_id" = CASE "source"::TEXT
              WHEN 'human' THEN 0 ELSE (CASE WHEN "source" IS NULL THEN 2 ELSE 1 END) END
        ''',
    })
    assert response.status == 500
    assert "exactly one statement" in response.text()
