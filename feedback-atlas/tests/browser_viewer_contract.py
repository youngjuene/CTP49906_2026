"""Real vendored desktop viewer contract, with GPU device acquisition held open.

The table, category setup, search and SQL connector are real. This does not test
GPU drawing; holding the device promise avoids pretending to implement WebGPU.
"""

import json

import pytest

pytest.register_assert_rewrite("browser_viewer")

from tests.auth_helpers import ACCESS_CODES

pytest_plugins = ["browser_viewer"]


@pytest.fixture
def context(browser):
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    context.add_init_script("""
        Object.defineProperty(navigator, 'gpu', {configurable:true, value:{
          wgslLanguageFeatures:new Set(),
          requestAdapter:async () => ({
            features:new Set(['shader-f16']),
            limits:{maxBufferSize:268435456, maxStorageBufferBindingSize:134217728},
            requestDevice:() => new Promise(() => {})
          })
        }});
    """)
    yield context
    context.close()


def open_viewer(live_server, context):
    page = context.new_page()
    queries, errors = [], []
    page.on("request", lambda request: queries.append(json.loads(request.post_data))
            if request.url.endswith("/data/query") and request.post_data else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(live_server.url)
    page.fill("#gate-code", ACCESS_CODES["writer1"])
    page.click("#gate-submit")
    page.wait_for_selector("#viewer-card:not([hidden])")
    return page, queries, errors


def test_real_viewer_initializes_with_only_read_queries(live_server, context):
    page, queries, errors = open_viewer(live_server, context)
    page.get_by_text("첫 의견입니다.", exact=True).first.wait_for(timeout=15_000)
    # A real category query must run: a mock that omits the embedding chart
    # would otherwise make a no-writes assertion vacuous.
    page.wait_for_function("""() => performance.getEntriesByType('resource')
      .filter(entry => entry.name.endsWith('/data/query')).length > 10""")
    assert any("CASE" in query["sql"] and "source" in query["sql"] for query in queries), queries
    assert all(query["type"] != "exec" for query in queries), queries
    assert not errors, errors


def is_category_count(response, field):
    if not response.url.endswith("/data/query") or not response.request.post_data:
        return False
    sql = json.loads(response.request.post_data)["sql"]
    return 'AS "index"' in sql and f'"{field}"' in sql


@pytest.mark.parametrize("field", ["week", "target_name", "timestamp", "x"])
def test_real_viewer_category_switches_remain_read_only(live_server, context, field):
    page, queries, errors = open_viewer(live_server, context)
    page.get_by_text("첫 의견입니다.", exact=True).first.wait_for(timeout=15_000)
    category = page.locator("#viewer-mount select").first
    with page.expect_response(lambda response: is_category_count(response, field)) as counted:
        category.select_option(json.dumps(field))
    assert counted.value.status == 200, counted.value.text()
    assert category.input_value() == json.dumps(field)
    assert page.is_visible("#viewer-card")
    assert all(query["type"] != "exec" for query in queries), queries
    assert not errors, errors


def test_real_viewer_refreshes_categories_and_table_without_reset(live_server, context):
    page, queries, errors = open_viewer(live_server, context)
    page.get_by_text("첫 의견입니다.", exact=True).first.wait_for(timeout=15_000)
    category = page.locator("#viewer-mount select").first
    assert category.input_value() == json.dumps("source")
    text = "새 AI 의견도 같은 분석 화면에 나타납니다."
    with page.expect_response(lambda response: is_category_count(response, "source")
                              and "'ai'" in json.loads(response.request.post_data)["sql"]) as updated:
        page.fill("#c-text", text)
        page.click("#c-submit")
    assert updated.value.status == 200
    page.get_by_text(text, exact=True).first.wait_for(timeout=15_000)
    assert category.input_value() == json.dumps("source")
    page.click("#tab-viewer")
    assert page.is_visible("#map-card")
    page.click("#tab-viewer")
    page.get_by_text(text, exact=True).first.wait_for(timeout=15_000)
    assert all(query["type"] != "exec" for query in queries), queries
    assert not errors, errors


def test_empty_real_viewer_accepts_its_first_opinion(live_server, context):
    # Clear only the isolated fixture database before restarting its owned server.
    live_server.stop()
    import sqlite3
    with sqlite3.connect(live_server.db) as db:
        db.execute("DELETE FROM opinions")
    live_server.start()
    page, queries, errors = open_viewer(live_server, context)
    page.locator("#viewer-mount select").first.wait_for(timeout=15_000)
    text = "빈 분석 화면에 들어오는 첫 의견"
    page.fill("#c-text", text)
    page.click("#c-submit")
    page.get_by_text(text, exact=True).first.wait_for(timeout=15_000)
    assert all(query["type"] != "exec" for query in queries), queries
    assert not errors, errors


def test_real_full_text_search_loads_worker_and_finds_new_data(live_server, context):
    from playwright.sync_api import expect

    page, queries, errors = open_viewer(live_server, context)
    page.get_by_text("첫 의견입니다.", exact=True).first.wait_for(timeout=15_000)
    with page.expect_response(lambda response: "search.worker-" in response.url) as worker:
        page.locator("#viewer-mount").get_by_role("button", name="Search", exact=True).click()
    assert worker.value.status == 200
    assert "/vendor/assets/" in worker.value.url
    search = page.locator("#viewer-mount input[type=search]")
    search.fill("첫")
    search.press("Enter")
    expect(page.get_by_text("1 result.", exact=True)).to_be_visible(timeout=10_000)
    text = "qaFreshOpinion 새로 등록한 의견"
    page.fill("#c-text", text)
    page.click("#c-submit")
    page.get_by_text(text, exact=True).first.wait_for(timeout=15_000)
    search.fill("qaFreshOpinion")
    search.press("Enter")
    expect(page.get_by_text("1 result.", exact=True)).to_be_visible(timeout=10_000)
    expect(page.get_by_text(text, exact=True)).to_have_count(2, timeout=10_000)  # table and search result
    assert all(query["type"] != "exec" for query in queries)
    assert not errors, errors


def test_real_viewer_crossfilter_survives_refresh_and_can_clear(live_server, context):
    from playwright.sync_api import expect

    page, queries, errors = open_viewer(live_server, context)
    page.get_by_text("첫 의견입니다.", exact=True).first.wait_for(timeout=15_000)
    page.get_by_title("o_browser_1", exact=True).click()
    expect(page.get_by_text("두 번째 의견입니다.", exact=True)).to_have_count(0)
    page.fill("#c-text", "필터 밖에서 새로 도착한 의견")
    page.click("#c-submit")
    page.wait_for_function("() => document.querySelector('#c-text').value === ''")
    expect(page.get_by_text("첫 의견입니다.", exact=True)).to_be_visible()
    expect(page.get_by_text("두 번째 의견입니다.", exact=True)).to_have_count(0)
    page.get_by_title("Clear filters", exact=True).click()
    expect(page.get_by_text("두 번째 의견입니다.", exact=True)).to_be_visible()
    expect(page.get_by_text("필터 밖에서 새로 도착한 의견", exact=True)).to_be_visible()
    assert all(query["type"] != "exec" for query in queries)
    assert not errors, errors


def test_real_viewer_groups_many_targets_in_other_category(live_server, context):
    import csv
    from src.models import Opinion
    from src.store import Store
    from src.textnorm import text_hash
    from tests.auth_helpers import write_access_codes_for

    live_server.stop()
    with live_server.roster.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "display_name", "role"])
        writer.writerow(["writer1", "Author", "observer"])
        writer.writerows([[f"target{i}", f"Group '{i}'", "student"] for i in range(12)])
    write_access_codes_for(live_server.tmpdir, ["writer1"] + [f"target{i}" for i in range(12)])
    store = Store(str(live_server.db))
    try:
        store.migrate()
        for i in range(12):
            opinion = Opinion(id=f"qa_many_{i}", reviewer_id="writer1", target_id=f"target{i}",
                              text=f"Many targets opinion {i}", source="human", week=2,
                              timestamp=f"2026-09-09T00:00:{i:02}.000Z")
            store.insert_opinion(opinion, text_hash(opinion.text))
    finally:
        store.close()
    live_server.start()
    page, queries, errors = open_viewer(live_server, context)
    page.locator("#viewer-mount select").first.wait_for(timeout=15_000)
    with page.expect_response(lambda response: response.url.endswith("/data/query")
                              and "otherCategoryCount" in (response.request.post_data or "")) as other:
        page.locator("#viewer-mount select").first.select_option(json.dumps("target_name"))
    assert other.value.status == 200, other.value.text()
    assert page.is_visible("#viewer-card")
    assert all(query["type"] != "exec" for query in queries)
    assert not errors, errors
