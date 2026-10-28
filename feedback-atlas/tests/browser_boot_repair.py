"""Focused boot recovery checks for the boot card.

Run: .venv/bin/python -m pytest tests/browser_boot_repair.py -q
"""

pytest_plugins = ["browser_viewer"]


def test_module_boot_failure_has_visible_retry(live_server, browser):
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
        page.goto(live_server.map_url, wait_until="networkidle")
        page.wait_for_selector("#boot-error:not([hidden])")
        assert page.is_visible("#boot-retry")
        assert page.is_hidden("#view-main")
        assert "앱을 시작하지 못했습니다" in page.inner_text("#boot-error-text")
        page.click("#boot-retry")
        page.wait_for_selector("#view-main:not([hidden])")
        page.wait_for_selector("#map [data-id]", state="attached")
    finally:
        context.close()


def test_a_fatal_participant_error_lands_on_the_boot_card(live_server, browser):
    """A page cached from an older release must say so rather than show a frozen map."""
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    try:
        def stale_app(route):
            body = route.fetch().text().replace("const PROTOCOL = 1;", "const PROTOCOL = 0;")
            route.fulfill(content_type="text/javascript", body=body)

        page.route("**/app.js", stale_app)
        page.goto(live_server.map_url)
        page.wait_for_selector("#boot-error:not([hidden])")
        assert page.is_hidden("#view-main")
        assert "새로고침" in page.inner_text("#boot-error-text")
    finally:
        context.close()
