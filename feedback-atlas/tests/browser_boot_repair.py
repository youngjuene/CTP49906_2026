"""Focused boot recovery checks for the inert landing page.

Run: .venv/bin/python -m pytest tests/browser_boot_repair.py -q
"""

from tests.auth_helpers import ACCESS_CODES

pytest_plugins = ["browser_viewer"]


def test_module_boot_failure_has_visible_retry_without_url_leak(live_server, browser):
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    failed_once = {"value": False}
    try:
        def app_route(route):
            if failed_once["value"]:
                route.continue_()
                return
            failed_once["value"] = True
            route.abort()

        page.route("**/app.js", app_route)
        page.goto(live_server.url, wait_until="networkidle")
        page.fill("#gate-code", ACCESS_CODES["writer1"])
        page.press("#gate-code", "Enter")
        page.wait_for_selector("#boot-retry:not([hidden])")
        assert page.is_disabled("#gate-submit")
        assert "앱을 시작하지 못했습니다" in page.inner_text("#gate-error")
        assert "code=" not in page.url, page.url
        assert ACCESS_CODES["writer1"] not in page.url
        page.click("#boot-retry")
        page.wait_for_selector("#gate-submit:not(:disabled)")
        page.fill("#gate-code", ACCESS_CODES["writer1"])
        page.click("#gate-submit")
        page.wait_for_selector("#view-main:not([hidden])")
    finally:
        context.close()


def test_bad_environment_json_falls_back_to_classroom_boot(live_server, browser):
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    try:
        page.route("**/api/environment", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body="{not-json",
        ))
        page.goto(live_server.url)
        page.wait_for_selector("#gate-submit:not(:disabled)")
        assert page.locator("#boot-retry:not([hidden])").count() == 0
        page.fill("#gate-code", ACCESS_CODES["writer1"])
        page.click("#gate-submit")
        page.wait_for_selector("#view-main:not([hidden])")
    finally:
        context.close()
