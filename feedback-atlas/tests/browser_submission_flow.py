"""V2 browser recovery tests against a temporary real server (fake model only)."""
import json
import pytest
from browser_classroom import enter, context
pytest_plugins = ["browser_viewer"]


def test_reload_restores_exact_draft_metadata_and_identity(live_server, context):
    page = context.new_page()
    enter(page, live_server.url)
    raw = '  조명 가 설명\n\n음악 🎵 의견  '
    page.fill('#c-text', raw)
    page.select_option('#c-target', 'target2')
    page.select_option('#c-week', '4')
    page.click('#c-source [data-value="human"]')
    page.get_by_text('이 기기에 저장됨', exact=True).wait_for()
    page.reload()
    page.wait_for_function("document.querySelector('#c-text').value.length > 0")
    assert page.input_value('#c-text') == raw
    assert page.input_value('#c-target') == 'target2'
    assert page.input_value('#c-week') == '4'
    assert page.get_attribute('#c-source [data-value="human"]', 'aria-pressed') == 'true'
    assert 'writer1' in page.inner_text('#own-id')
    page.click('#change-id')
    page.fill('#gate-id', 'target1')
    page.click('#gate-submit')
    page.wait_for_selector('#view-main:not([hidden])')
    assert page.input_value('#c-text') == ''


def test_long_raw_send_persists_envelope_before_wire_and_ack_keeps_new_draft(live_server, context):
    context.add_init_script('''
      const Real = WebSocket;
      window.sentFrames = [];
      window.WebSocket = class extends Real {
        send(value) { const f = JSON.parse(value); window.sentFrames.push(f); super.send(value); }
        set onmessage(fn) { super.onmessage = e => {
          if(JSON.parse(e.data).t === 'submission_accepted') setTimeout(() => fn(e), 900);
          else fn(e);
        }; }
      };
    ''')
    page = context.new_page()
    enter(page, live_server.url)
    raw = '  '+ '한글 🎵 가 문장입니다. ' * 100 + '\n  '
    page.select_option('#c-target', 'target2')
    page.fill('#c-text', raw)
    assert page.get_attribute('#c-text', 'maxlength') is None
    page.click('#c-submit')
    page.wait_for_function("window.sentFrames.some(f=>f.t==='submit')")
    sent = page.evaluate("window.sentFrames.find(f=>f.t==='submit')")
    assert sent['text'] == raw
    assert len(sent['owner_capability']) == 43
    assert sent['source'] == 'ai'
    page.fill('#c-text', '새로 쓰는 다음 의견')
    page.get_by_text('원문 저장됨', exact=False).first.wait_for()
    assert page.input_value('#c-text') == '새로 쓰는 다음 의견'
    page.reload()
    page.wait_for_function("document.querySelector('#c-text').value === '새로 쓰는 다음 의견'")


def test_storage_failure_never_claims_saved_or_sends(live_server, context):
    context.add_init_script("Object.defineProperty(window, 'indexedDB', {get(){throw Error('denied')}})")
    page = context.new_page()
    enter(page, live_server.url)
    page.select_option('#c-target', 'target1')
    page.fill('#c-text', '저장 실패에도 원문 보존')
    page.click('#c-submit')
    page.get_by_text('기기 저장을 사용할 수 없습니다', exact=False).first.wait_for()
    assert page.input_value('#c-text') == '저장 실패에도 원문 보존'
    assert page.locator('.submission-card').count() == 0


def test_lost_ack_reload_replays_one_persisted_envelope(live_server, context):
    context.add_init_script('''
      const Real = WebSocket;
      window.sentFrames = [];
      window.WebSocket = class extends Real {
        send(value){ window.sentFrames.push(JSON.parse(value)); super.send(value); }
        set onmessage(fn){ super.onmessage = e => {
          if(JSON.parse(e.data).t === 'submission_accepted' && !sessionStorage.dropped){
            sessionStorage.dropped = '1'; window.ackDropped = true; this.close(); return;
          } fn(e);
        }; }
      };
    ''')
    page = context.new_page()
    enter(page, live_server.url)
    page.select_option('#c-target', 'target2')
    page.fill('#c-text', '응답 유실 원문 🎵')
    page.click('#c-submit')
    page.wait_for_function('window.ackDropped')
    original = page.evaluate("window.sentFrames.find(f=>f.t==='submit')")
    page.reload()
    page.wait_for_function("document.querySelector('#c-text').value === ''")
    page.locator('.submission-card').first.wait_for()
    page.wait_for_function("window.sentFrames.some(f=>f.t==='submit' || f.t==='submission_receipt')")
    retries = page.evaluate("window.sentFrames.filter(f=>f.t==='submit')")
    for retry in retries:
        assert retry['nonce'] == original['nonce']
        assert retry['owner_capability'] == original['owner_capability']
        assert retry['text'] == original['text']


def test_duplicate_tabs_claim_one_lease_and_ack_cannot_clear_new_version(live_server, context):
    page, twin = context.new_page(), context.new_page()
    enter(page, live_server.url)
    enter(twin, live_server.url)
    page.evaluate('''async()=>{
      const s=await import('/draft-store.js');
      const draft={key:'qa-atomic',scope:'qa-atomic',version:'v1',text:'첫 원문'};
      await s.enqueue(draft,{nonce:'qa-one',owner_capability:s.randomToken(),text:draft.text});
      await s.saveDraft({...draft,version:'v2',text:'새 원문'});
    }''')
    first=page.evaluate("async()=> (await import('/draft-store.js')).claimLease('qa-atomic:qa-one','tab-one')")
    second=twin.evaluate("async()=> (await import('/draft-store.js')).claimLease('qa-atomic:qa-one','tab-two')")
    assert first is True and second is False
    result=page.evaluate('''async()=>{
      const s=await import('/draft-store.js'), saved=await s.readScope('qa-atomic');
      await s.acknowledge(saved.outbox[0],{submission_id:'qa-parent',nonce:'qa-one',revision:1,state:'queued'});
      return s.readScope('qa-atomic');
    }''')
    assert result['drafts'][0]['text'] == '새 원문'
    assert result['outbox'] == []
    assert len(result['receipts']) == 1


def test_dirty_context_keeps_metadata_until_explicit_choice(live_server, context):
    from browser_export_scopes import admin
    writer, instructor=context.new_page(),context.new_page()
    enter(writer, live_server.url)
    writer.select_option('#c-target','target1')
    writer.select_option('#c-week','2')
    writer.fill('#c-text','아직 작성 중인 원문')
    admin(instructor,live_server.url)
    instructor.select_option('#class-target','target2')
    instructor.select_option('#class-week','4')
    instructor.click('#class-save')
    writer.wait_for_selector('#c-context:not([hidden])')
    assert writer.input_value('#c-target') == 'target1'
    assert writer.input_value('#c-week') == '2'
    writer.click('#context-keep')
    writer.wait_for_selector('#c-context',state='hidden')
    assert writer.input_value('#c-target') == 'target1'
    assert writer.input_value('#c-text') == '아직 작성 중인 원문'


def test_second_tab_cannot_overwrite_a_newer_persisted_draft(live_server, context):
    page=context.new_page()
    enter(page,live_server.url)
    result=page.evaluate('''async()=>{
      const s=await import('/draft-store.js');
      await s.saveDraft({key:'qa-conflict',scope:'qa-conflict',version:'base',text:'공통 초안'});
      await s.saveDraft({key:'qa-conflict',scope:'qa-conflict',version:'tab1',text:'첫 탭 수정'},'base');
      let conflict=false;
      try {await s.saveDraft({key:'qa-conflict',scope:'qa-conflict',version:'tab2',text:'둘째 탭 수정'},'base');}
      catch(e){conflict=e.code==='DRAFT_CONFLICT';}
      return {conflict,saved:(await s.readScope('qa-conflict')).drafts[0].text};
    }''')
    assert result == {'conflict':True,'saved':'첫 탭 수정'}


def test_only_one_tab_can_enqueue_the_same_shared_draft(live_server, context):
    page=context.new_page()
    enter(page,live_server.url)
    result=page.evaluate('''async()=>{
      const s=await import('/draft-store.js');
      const draft={key:'qa-send-race',scope:'qa-send-race',version:'base',text:'원문 하나'};
      await s.saveDraft(draft);
      const outcomes=await Promise.allSettled([
        s.enqueue(draft,{nonce:'first',owner_capability:s.randomToken(),text:draft.text}),
        s.enqueue(draft,{nonce:'second',owner_capability:s.randomToken(),text:draft.text})
      ]);
      return {accepted:outcomes.filter(r=>r.status==='fulfilled').length,
        queued:(await s.readScope('qa-send-race')).outbox.length};
    }''')
    assert result == {'accepted':1,'queued':1}


def test_confirm_rejected_original_preserves_newer_draft(live_server, context):
    from browser_export_scopes import admin
    context.add_init_script('''
      const Real=WebSocket;
      window.WebSocket=class extends Real {
        send(value){const f=JSON.parse(value);
          if(f.t==='submit' && !window.delayedOnce){window.delayedOnce=true;window.releaseSubmit=()=>super.send(value);return;}
          super.send(value);
        }
      };
    ''')
    page,instructor=context.new_page(),context.new_page()
    enter(page,live_server.url)
    admin(instructor,live_server.url)
    page.select_option('#c-target','target1')
    page.fill('#c-text','ORIGINAL_A_NOT_ACCEPTED')
    page.click('#c-submit')
    page.wait_for_function('Boolean(window.releaseSubmit)')
    page.fill('#c-text','NEW_DRAFT_B')
    instructor.select_option('#class-target','target2')
    instructor.select_option('#class-week','4')
    instructor.click('#class-save')
    page.wait_for_selector('#c-context:not([hidden])')
    page.evaluate('window.releaseSubmit()')
    page.wait_for_function("document.querySelector('#c-hint').textContent.includes('발표 대상 또는 주차가 변경되었습니다.')")
    page.wait_for_function("!document.querySelector('#c-submit').disabled")
    page.click('#context-keep')
    page.locator('.submission-card',has_text='반영됨').first.wait_for()
    assert page.input_value('#c-text')=='NEW_DRAFT_B'
    page.locator('.submission-card button',has_text='확인·수정').first.click()
    page.wait_for_function("document.querySelector('#reader-original').textContent==='ORIGINAL_A_NOT_ACCEPTED'")
    assert page.input_value('#reader-target')=='target1'
    page.keyboard.press('Escape')
    page.reload()
    page.wait_for_function("document.querySelector('#c-text').value==='NEW_DRAFT_B'")


def protocol_probe(context):
    context.add_init_script('''
      const Real=WebSocket;window.protocolSockets=[];window.protocolFrames=[];
      window.WebSocket=class extends Real {
        constructor(...args){super(...args);window.protocolSockets.push(this);this.addEventListener('message',e=>window.protocolFrames.push(JSON.parse(e.data)));}
        send(value){const f=JSON.parse(value);
          if(f.t==='hello' && window.forceWrongProtocol)f.protocol=1;
          if(f.t==='submit' && window.holdProtocolSubmit){window.heldProtocolSubmit=f;return;}
          super.send(JSON.stringify(f));
        }
      };
    ''')


def test_protocol_mismatch_preserves_outbox_and_new_draft_until_explicit_refresh(live_server, context):
    protocol_probe(context)
    page=context.new_page()
    enter(page,live_server.url)
    page.select_option('#c-target','target2')
    page.select_option('#c-week','4')
    page.click('#c-source [data-value="human"]')
    original='  먼저 보낸 원문 A 🎵  '
    newer='새로 적던 원문 B 가\n\n보존합니다.'
    page.fill('#c-text',original)
    page.evaluate('window.holdProtocolSubmit=true')
    page.click('#c-submit')
    page.wait_for_function('Boolean(window.heldProtocolSubmit)')
    page.fill('#c-text',newer)
    page.evaluate('window.forceWrongProtocol=true;window.protocolSockets.at(-1).close()')
    page.wait_for_function("document.querySelector('#gate-error-text').textContent.includes('버전')")
    assert page.locator('#protocol-refresh').count()==1
    page.wait_for_function("!document.querySelector('#protocol-refresh').disabled")
    saved=page.evaluate('''async()=>{
      const s=await import('/draft-store.js'),hello=window.protocolFrames.find(f=>f.t==='hello_ok');
      return s.readScope(s.draftKey(hello.dataset_id,hello.submitter_key));
    }''')
    assert saved['drafts'][0]['text']==newer
    assert saved['outbox'][0]['frame']['text']==original
    assert saved['outbox'][0]['frame']['nonce']==page.evaluate('window.heldProtocolSubmit.nonce')
    with page.expect_download() as result:
        page.click('#protocol-download')
    assert result.value.path().read_text(encoding='utf-8')==newer
    assert page.is_visible('#view-gate')
    page.click('#protocol-refresh')
    page.wait_for_function("document.querySelector('#c-text').value.includes('새로 적던 원문 B')")
    assert page.input_value('#c-text')==newer
    assert page.input_value('#c-target')=='target2'
    assert page.input_value('#c-week')=='4'
    page.locator('.submission-card',has_text='먼저 보낸 원문 A').filter(has_text='반영됨').wait_for(timeout=25000)
    assert page.input_value('#c-text')==newer


def test_protocol_mismatch_requires_download_when_storage_is_unavailable(live_server, context):
    protocol_probe(context)
    context.add_init_script("Object.defineProperty(window,'indexedDB',{get(){throw Error('denied')}})")
    page=context.new_page()
    enter(page,live_server.url)
    raw='기기에 저장할 수 없는 원문도 내려받습니다. 🎧'
    page.fill('#c-text',raw)
    page.evaluate('window.forceWrongProtocol=true;window.protocolSockets.at(-1).close()')
    page.wait_for_function("document.querySelector('#gate-error-text').textContent.includes('버전')")
    assert page.locator('#protocol-refresh').count()==1
    page.wait_for_function("document.querySelector('#protocol-status').textContent.includes('내려받')")
    assert page.is_disabled('#protocol-refresh')
    assert page.input_value('#c-text')==raw
    with page.expect_download() as result:
        page.click('#protocol-download')
    assert result.value.path().read_text(encoding='utf-8')==raw
    assert page.is_enabled('#protocol-refresh')
    assert page.is_visible('#view-gate')
