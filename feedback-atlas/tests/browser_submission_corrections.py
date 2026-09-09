"""Owner corrections and public original reading."""
import pytest
from browser_classroom import enter, context
pytest_plugins = ["browser_viewer"]


def test_owner_split_source_history_and_withdraw(live_server, context):
    page = context.new_page()
    enter(page, live_server.url)
    raw = '한글 🎵 첫 의견입니다. 두 번째 의견입니다.'
    page.select_option('#c-target', 'target2')
    page.fill('#c-text', raw)
    page.click('#c-submit')
    page.locator('.submission-card', has_text='반영됨').first.wait_for()
    page.locator('.submission-card button', has_text='확인·수정').first.click()
    page.wait_for_selector('#feedback-reader[open]')
    assert page.inner_text('#reader-original') == raw
    initial_units = page.locator('.unit-source').count()
    page.locator('.unit-text').first.focus()
    page.keyboard.press('Control+Home')
    for _ in range(6):
        page.keyboard.press('ArrowRight')
    caret = page.locator('.unit-text').first.evaluate('(el)=>({start:el.selectionStart,end:el.selectionEnd,value:el.value,focused:document.activeElement===el})')
    page.locator('.unit-split').first.click()
    assert page.locator('.unit-source').count() == initial_units + 1, {'caret':caret,'message':page.inner_text('#reader-message')}
    page.locator('.unit-source').last.select_option('human')
    page.click('#reader-save')
    page.wait_for_function("document.querySelector('#reader-history').textContent.includes('2')")
    assert '"start"' not in page.locator('#reader-history').text_content()
    assert '사람 작성' in page.locator('#reader-history').text_content()
    page.click('#reader-withdraw')
    page.click('#reader-confirm-withdraw')
    page.get_by_text('철회됨', exact=False).first.wait_for()
    page.keyboard.press('Escape')
    assert not page.locator('#feedback-reader').is_visible()


def test_keyboard_public_reader_returns_focus(live_server, context):
    page = context.new_page()
    enter(page, live_server.url)
    page.locator('.result-browser summary').click()
    button = page.locator('#visible-results button').first
    button.focus()
    page.keyboard.press('Enter')
    page.wait_for_selector('#feedback-reader[open]')
    assert page.inner_text('#reader-original')
    assert page.locator('#reader-save').is_hidden()
    page.keyboard.press('Escape')
    assert button.evaluate('(el)=>el === document.activeElement')


def test_touch_tap_reads_original_and_escape_returns_to_map(live_server, browser):
    context=browser.new_context(viewport={'width':390,'height':844},has_touch=True)
    page=context.new_page()
    try:
        enter(page,live_server.url)
        page.click('#tab-legend')
        page.locator('#map [data-id]').first.tap(force=True)
        page.wait_for_selector('#feedback-reader[open]')
        page.wait_for_function("document.querySelector('#reader-original').textContent.length > 0")
        page.keyboard.press('Escape')
        page.wait_for_function("document.activeElement?.id === 'map'")
        assert page.locator('#map').evaluate('(el)=>el === document.activeElement'), page.evaluate('document.activeElement.outerHTML')
    finally:
        context.close()


def test_unknown_correction_result_blocks_withdraw_until_explicit_retry(live_server, context):
    context.add_init_script('''
      const Real=WebSocket;window.mutations=[];window.correctionRequests=new Set();
      window.WebSocket=class extends Real {
        send(value){const f=JSON.parse(value);if(f.t==='submission_correct'||f.t==='submission_withdraw'){window.mutations.push(f);if(f.t==='submission_correct')window.correctionRequests.add(f.request_id);}super.send(value);}
        set onmessage(fn){super.onmessage=e=>{const f=JSON.parse(e.data);
          if(f.t==='submission_detail' && window.correctionRequests.has(f.request_id) && !window.responseDropped){window.responseDropped=true;return;}fn(e);
        };}
      };
    ''')
    page=context.new_page()
    enter(page,live_server.url)
    page.select_option('#c-target','target1')
    page.fill('#c-text','수정 응답을 잃어도 철회 요청으로 바꾸지 않습니다.')
    page.click('#c-submit')
    page.locator('.submission-card',has_text='반영됨').first.wait_for()
    page.locator('.submission-card button',has_text='확인·수정').first.click()
    page.locator('.unit-source').first.select_option('human')
    page.click('#reader-save')
    page.wait_for_selector('#reader-resolve:not([hidden])',timeout=15000)
    assert page.is_disabled('#reader-withdraw')
    assert page.is_disabled('#reader-save')
    page.click('#reader-resolve')
    page.wait_for_selector('#reader-resolve',state='hidden')
    messages=page.evaluate('window.mutations')
    assert [m['t'] for m in messages]==['submission_correct','submission_correct']
    assert messages[0]['action_nonce']==messages[1]['action_nonce']
    page.click('#reader-withdraw')
    page.click('#reader-confirm-withdraw')
    page.get_by_text('철회됨',exact=False).first.wait_for()
    assert page.evaluate("window.mutations.at(-1).t")=='submission_withdraw'


def test_late_correction_response_cannot_replace_another_open_original(live_server, context):
    context.add_init_script('''
      const Real=WebSocket;window.correctionIds=new Set();
      window.WebSocket=class extends Real {
        send(value){const f=JSON.parse(value);if(f.t==='submission_correct')window.correctionIds.add(f.request_id);super.send(value);}
        set onmessage(fn){super.onmessage=e=>{const f=JSON.parse(e.data);
          if(f.t==='submission_detail' && window.correctionIds.has(f.request_id) && !window.heldCorrection){window.heldCorrection=true;window.releaseCorrection=()=>fn(e);return;}fn(e);
        };}
      };
    ''')
    page=context.new_page()
    enter(page,live_server.url)
    page.select_option('#c-target','target1')
    for raw in ('ORIGINAL_A_FOR_LATE_REPLY','ORIGINAL_B_STAYS_OPEN'):
        page.fill('#c-text',raw)
        page.click('#c-submit')
        page.locator('.submission-card',has_text=raw).filter(has_text='반영됨').wait_for()
    page.locator('.submission-card',has_text='ORIGINAL_A_FOR_LATE_REPLY').get_by_text('확인·수정').click()
    page.locator('.unit-source').first.select_option('human')
    page.click('#reader-save')
    page.wait_for_function('Boolean(window.releaseCorrection)')
    page.keyboard.press('Escape')
    page.locator('.submission-card',has_text='ORIGINAL_B_STAYS_OPEN').get_by_text('확인·수정').click()
    page.locator('.unit-source').first.select_option('human')
    page.evaluate('window.releaseCorrection()')
    page.wait_for_timeout(150)
    assert page.inner_text('#reader-original')=='ORIGINAL_B_STAYS_OPEN'
    assert page.locator('.unit-source').first.input_value()=='human'


def test_caret_moves_over_complete_emoji_grapheme(live_server, context):
    page=context.new_page()
    enter(page,live_server.url)
    page.select_option('#c-target','target1')
    page.fill('#c-text','가 👩🏽‍💻 끝 설명입니다.')
    page.click('#c-submit')
    page.locator('.submission-card',has_text='반영됨').first.wait_for()
    page.locator('.submission-card button',has_text='확인·수정').first.click()
    page.locator('.unit-text').first.focus()
    page.keyboard.press('Control+Home')
    for _ in range(3):
        page.keyboard.press('ArrowRight')
    page.locator('.unit-split').first.click()
    assert page.locator('.unit-text').first.input_value()=='가 👩🏽‍💻'
    assert page.locator('.unit-text').nth(1).input_value()==' 끝 설명입니다.'
