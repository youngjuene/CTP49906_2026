"""Visible / selected / all scopes use the displayed published revision."""
import csv
import io
import pytest
from browser_viewer import CODE
from browser_classroom import context
pytest_plugins = ["browser_viewer"]


def admin(page, url):
    page.goto(url+'#admin')
    page.fill('#admin-code', CODE)
    page.locator('#admin-form button[type="submit"]').click()
    page.wait_for_selector('#adminpanel:not([hidden])')
    page.wait_for_selector('#map [data-id]')


def test_visible_scope_follows_search_and_empty_selected_stays_empty(live_server, context):
    page = context.new_page()
    admin(page, live_server.url)
    assert page.inner_text('#admin-counts') == '전체 의견 3개 · 원문 3건 · 제출자 1명'
    page.fill('#corpus-search', '첫 의견')
    assert '1' in page.inner_text('#export-preview')
    assert page.locator('#visible-results button').count() == 1
    with page.expect_download() as result:
        page.click('#btn-export')
    rows = list(csv.DictReader(io.StringIO(result.value.path().read_text(encoding='utf-8-sig'))))
    assert [r['id'] for r in rows] == ['o_browser_1']
    page.select_option('#export-scope', 'selected')
    assert '0' in page.inner_text('#export-preview')
    assert page.is_disabled('#btn-export')
    page.select_option('#export-scope', 'all')
    assert '3' in page.inner_text('#export-preview')


def test_revision_conflict_requires_second_export_click(live_server, context):
    page=context.new_page()
    admin(page,live_server.url)
    calls=[]
    def intercept(route):
        calls.append(route.request.post_data_json)
        if len(calls)==1:
            route.fulfill(status=409,content_type='application/json',body='{"error":"stale revision"}')
        else:
            route.continue_()
    # Only fault injection: no successful fake export or successful fake server response.
    page.route('**/api/export.csv',intercept)
    page.click('#btn-export')
    page.get_by_text('데이터가 바뀌었습니다.',exact=False).wait_for()
    assert len(calls)==1
    with page.expect_download():
        page.click('#btn-export')
    assert len(calls)==2
    assert calls[1]['scope']=='visible'
