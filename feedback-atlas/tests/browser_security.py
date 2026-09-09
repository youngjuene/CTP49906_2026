"""Real-browser security flows with CSP enabled and isolated test data."""

import pytest

from browser_viewer import CODE
from tests.auth_helpers import ACCESS_CODES

pytest_plugins = ["browser_viewer"]


def test_legacy_admin_bookmark_redirects_to_path_and_back_to_classroom(live_server, context):
    page = context.new_page()
    page.goto(live_server.url + "#admin")
    page.wait_for_url(live_server.url + "admin")
    page.wait_for_selector("#view-admin-gate:not([hidden])")
    page.click("#admin-back")
    page.wait_for_url(live_server.url)
    page.wait_for_selector("#view-gate:not([hidden])")


@pytest.fixture
def context(browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844})
    yield ctx
    ctx.close()


def test_private_code_login_and_session_restore_with_csp(live_server, context):
    page = context.new_page()
    failures = []
    page.on("pageerror", lambda exc: failures.append(str(exc)))
    response = page.goto(live_server.url)
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    page.click("#gate-submit")
    assert "개인 접근 코드" in page.inner_text("#gate-error-text")
    assert page.locator("#gate-id").count() == 0
    assert page.locator('#gate-code[type="password"]').count() == 1
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")
    assert page.input_value("#gate-code") == ""
    storage = page.evaluate("JSON.stringify({...sessionStorage, ...localStorage})")
    assert ACCESS_CODES["writer1"] not in storage
    assert "writer1" not in storage
    page.reload()
    page.wait_for_selector("#view-main:not([hidden])")
    assert page.locator("#map [data-id]").count() == 3
    assert not failures, failures


def test_unauthenticated_browser_cannot_query_feedback(live_server, context):
    page = context.new_page()
    page.goto(live_server.url)
    result = page.evaluate("""async () => {
        const r = await fetch('/data/query', {method:'POST',
          headers:{'content-type':'application/json'},
          body:JSON.stringify({type:'json',sql:'SELECT text FROM dataset'})});
        return {status:r.status, body:await r.json()};
    }""")
    assert result["status"] == 401
    assert result["body"]["code"] == "NOT_AUTHENTICATED"


def test_admin_then_participant_has_no_cached_authorship(live_server, context):
    page = context.new_page()
    page.goto(live_server.url + "admin")
    page.fill("#admin-code", CODE)
    page.locator("#admin-form button[type=submit]").click()
    page.wait_for_selector("#view-main:not([hidden])")
    page.goto(live_server.url)
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    page.click("#gate-submit")
    page.wait_for_selector("#view-main:not([hidden])")
    assert page.locator("#adminpanel").is_hidden()
    assert page.locator("#admin-badge").is_hidden()
