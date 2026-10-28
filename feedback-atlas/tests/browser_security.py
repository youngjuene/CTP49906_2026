"""Real-browser security flows with CSP enabled and isolated test data."""

import pytest

from browser_viewer import CODE

pytest_plugins = ["browser_viewer"]


def test_legacy_admin_bookmark_redirects_to_path_and_back_to_classroom(live_server, context):
    page = context.new_page()
    page.goto(live_server.url + "#admin")
    page.wait_for_url(live_server.url + "admin")
    page.wait_for_selector("#view-admin-gate:not([hidden])")
    page.click("#admin-back")
    page.wait_for_url(live_server.map_url)
    page.wait_for_selector("#view-main:not([hidden])")


@pytest.fixture
def context(browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844})
    yield ctx
    ctx.close()


def test_anonymous_boot_and_reload_with_csp(live_server, context):
    page = context.new_page()
    failures = []
    page.on("pageerror", lambda exc: failures.append(str(exc)))
    response = page.goto(live_server.map_url)
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")
    assert page.locator("input[type=password]:visible").count() == 0
    storage = page.evaluate("JSON.stringify({...sessionStorage, ...localStorage})")
    assert "writer1" not in storage and "작성자1" not in storage
    page.reload()
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")
    assert page.locator("#map [data-id]").count() == 3
    assert not failures, failures


def test_anonymous_browser_reads_feedback_but_never_authorship(live_server, context):
    page = context.new_page()
    page.goto(live_server.map_url)
    result = page.evaluate("""async () => {
        const ask = async (sql) => {
          const r = await fetch('/data/query', {method:'POST',
            headers:{'content-type':'application/json'},
            body:JSON.stringify({type:'json', sql})});
          return {status:r.status, body:await r.json()};
        };
        return [await ask('SELECT text FROM dataset'),
                await ask('SELECT reviewer_id FROM dataset')];
    }""")
    assert result[0]["status"] == 200 and len(result[0]["body"]) == 3
    assert result[1]["status"] == 500
    assert "reviewer_id" in result[1]["body"]["error"]


def test_admin_then_participant_has_no_cached_authorship(live_server, context):
    page = context.new_page()
    page.goto(live_server.url + "admin")
    page.fill("#admin-code", CODE)
    page.locator("#admin-form button[type=submit]").click()
    page.wait_for_selector("#view-main:not([hidden])")
    page.goto(live_server.map_url)
    page.wait_for_selector("#view-main:not([hidden])")
    page.wait_for_selector("#map [data-id]", state="attached")
    assert page.locator("#adminpanel").is_hidden()
    assert page.locator("#admin-badge").is_hidden()
    assert "작성자1" not in page.content()
