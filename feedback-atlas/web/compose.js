import * as store from './draft-store.js';

export function mountCompose({state, request}) {
  const $ = id => document.getElementById(id);
  const text = $('c-text'), target = $('c-target'), week = $('c-week'), seg = $('c-source');
  let scope = null, draft = null, timer, loading = false, busy = false, generation = 0, ready = false, blocked = false;
  const leaseOwner = store.newVersion();
  let savedVersion=null, saveChain=Promise.resolve();
  const hint = message => { $('c-hint').textContent = message; };
  const storageError = error => {$('c-local').textContent = ['DRAFT_CONFLICT','QUEUE_BUSY'].includes(error?.code) ? error.message : '기기 저장을 사용할 수 없습니다. 새로고침 전에 원문을 내려받아 주세요.';};
  let receiptSignature='';
  const emitReceipts = rows => {const signature=JSON.stringify(rows);if(signature===receiptSignature)return;receiptSignature=signature;document.dispatchEvent(new CustomEvent('atlas:receipts', {detail:rows}));};
  function takeDraft() {
    return {...draft, key:scope, scope, text:text.value, target_id:target.value, week:Number(week.value)};
  }
  function render() {
    if (!draft) return;
    for (const b of seg.children) b.setAttribute('aria-pressed', String(b.dataset.value === draft.source));
    const length = Array.from(text.value).length;
    $('c-count').textContent = `${length.toLocaleString()} / ${state.defaults.maxChars.toLocaleString()}자`;
    $('c-summary').textContent = `${target.selectedOptions[0]?.textContent || '대상 선택 필요'} · ${week.value}주차 · ${draft.source === 'ai' ? 'AI 생성' : '사람 작성'}`;
    $('c-submit').disabled = !ready || busy || length > state.defaults.maxChars;
    if (length > state.defaults.maxChars) hint(`최대 ${state.defaults.maxChars.toLocaleString()}자입니다. 원문은 잘리지 않고 보관됩니다.`);
    $('c-context').hidden = !blocked && (draft.context_revision === state.context?.revision || draft.metadata_confirmed);
    $('c-closed').hidden = state.context?.accepting !== false;
  }
  async function persist() {
    clearTimeout(timer);
    if (!scope || !draft || loading) return;
    const current = takeDraft(), gen = generation;
    const task=saveChain.catch(()=>{}).then(async()=>{
      if(gen !== generation)return;
      await store.saveDraft(current,savedVersion);
      if(gen===generation){savedVersion=current.version;$('c-local').textContent='이 기기에 저장됨';}
    });
    saveChain=task;
    try{await task;}catch(error){if(gen===generation)storageError(error);throw error;}
  }
  function changed() {
    if (!draft || loading) return;
    draft = {...draft, dirty:true, version:store.newVersion()};
    $('c-local').textContent = '기기에 저장 중…';
    clearTimeout(timer); timer = setTimeout(() => persist().catch(() => {}), 250);
    render();
  }
  function fill(current) {
    draft = current;
    text.value = current.text;
    target.value = current.target_id || '';
    week.value = String(current.week);
    render();
  }
  function defaultDraft() {
    return {key:scope, scope, text:'', target_id:state.context?.target_id || '', week:state.context?.week || state.defaults.week,
      source:'ai', context_revision:state.context?.revision || 0, metadata_confirmed:false, dirty:false, version:store.newVersion()};
  }
  async function refreshReceipts() {
    const gen = generation, saved = await store.readScope(scope);
    if(gen === generation) emitReceipts(saved.receipts);
    return saved;
  }
  async function accepted(item, receipt) {
    const gen = generation;
    // Save whatever was typed during the request before the atomic version check.
    try{await persist();}catch(error){if(error.code !== 'DRAFT_CONFLICT')throw error;}
    const cleared = await store.acknowledge(item, receipt);
    if(gen !== generation) return;
    if(cleared) savedVersion=cleared.version;
    if (draft.version === item.version && cleared) fill(cleared);
    $('c-pending-context').textContent='';
    hint('원문 저장됨. 처리 결과는 아래에서 확인할 수 있습니다.');
    await refreshReceipts();
  }
  async function drain() {
    if (!ready || !scope || busy || !state.connected || state.channel === 'admin') return;
    const gen = generation;
    busy = true; render();
    try {
      const saved = await refreshReceipts();
      for (const item of saved.outbox) {
        if(gen !== generation) break;
        if (item.blocked) {blocked=true; $('c-pending-context').textContent=`전송 대기 원문: ${item.frame.text.slice(0,100)} · ${item.frame.target_id} · ${item.frame.week}주차. 아래 선택은 이 전송 대기 원문에 적용됩니다.`; hint('전송 대기: 수업 정보 확인이 필요합니다. 아래에서 현재 정보 적용 또는 기존 정보 유지를 선택하세요.'); $('c-context').hidden=false; continue;}
        if (!(await store.claimLease(item.key, leaseOwner))) continue;
        try {
          $('c-local').textContent = '전송 대기';
          const receipt = await request(item.frame);
          if(gen !== generation) break;
          await accepted(item, receipt);
        } catch (error) {
          if(gen !== generation) break;
          if (['STALE_CONTEXT','CLASS_CLOSED','TEXT_TOO_LONG','UNKNOWN_TARGET','BAD_WEEK','BAD_SOURCE','MALFORMED','NONCE_CONFLICT'].includes(error.code)) {
            blocked=true; $('c-pending-context').textContent=`전송 대기 원문: ${item.frame.text.slice(0,100)} · ${item.frame.target_id} · ${item.frame.week}주차. 아래 선택은 이 전송 대기 원문에 적용됩니다.`; await store.updateOutbox({...item, blocked:error.code});
            $('c-context').hidden=false;
          }
          hint(`${error.message || '연결을 확인하고 다시 시도합니다.'} 원문은 전송 대기에 보관됩니다.`);
        }
      }
    } catch { storageError(); }
    finally {busy = false; render();}
  }
  async function initialize() {
    const admin = state.channel === 'admin';
    $('compose').hidden = admin; $('tab-compose').hidden = admin;
    if(admin) return;
    const nextScope = store.draftKey(state.datasetId, state.submitterKey || state.identity);
    const sameScope = scope === nextScope && draft;
    const existing = sameScope ? takeDraft() : null;
    scope=nextScope; const gen=++generation; ready=false; loading=true;text.disabled=true;blocked=false;
    target.replaceChildren(new Option('대상 학생을 선택하세요', ''));
    state.targets.forEach(t => target.append(new Option(t.display_name, t.id)));
    week.replaceChildren(...[1,2,3,4].map(w => new Option(`${w}주차`, String(w))));
    if(!sameScope) {fill(defaultDraft());$('c-pending-context').textContent='';hint('보내기 전에 대상·주차·출처를 확인하세요.');}
    try {
      const saved = await store.readScope(nextScope);
      if(gen!==generation)return;
      if(!sameScope) savedVersion=saved.drafts[0]?.version || null;
      fill(existing || saved.drafts[0] || defaultDraft());
      emitReceipts(saved.receipts);
      $('c-local').textContent = saved.drafts.length ? '이 기기에 저장됨' : '';
    } catch {if(gen!==generation)return;fill(existing || defaultDraft()); storageError();}
    loading=false; ready=true;text.disabled=false;
    contextChanged(); render(); drain();
  }
  function contextChanged() {
    if(!draft) return;
    if(draft.context_revision !== state.context.revision) draft.metadata_confirmed=false;
    if (!draft.dirty && !text.value) {
      draft={...draft, target_id:state.context.target_id || '', week:state.context.week,
        context_revision:state.context.revision, metadata_confirmed:false};
      fill(draft);
    }
    render();
  }
  async function confirmContext(apply) {
    if (busy) {hint('전송 결과를 확인 중입니다. 잠시 후 다시 선택하세요.'); return;}
    const gen=generation, originalScope=scope;
    try {
      const saved = await store.readScope(originalScope);
      if(gen!==generation)return;
      for (const item of saved.outbox) {
        if(!item.blocked) {hint('전송 여부를 먼저 확인하고 있습니다. 연결 후 다시 선택하세요.'); return;}
        if(apply && !state.context.target_id) {hint('현재 발표 대상이 없습니다. 기존 정보 유지 또는 강사의 발표 대상 설정 후 다시 확인해 주세요.');return;}
        const result = await request({t:'submission_receipt', nonce:item.frame.nonce, owner_capability:item.frame.owner_capability});
        if(gen!==generation)return;
        if(result.receipt) await accepted(item, result.receipt);
        else {
          const frame={...item.frame,nonce:store.newVersion(),owner_capability:store.randomToken(),
            target_id:apply?state.context.target_id:item.frame.target_id,
            week:apply?state.context.week:item.frame.week,
            context_revision:state.context.revision,metadata_confirmed:true};
          // Only the rejected envelope is replaced. Its original A is never
          // reconstructed from a newer composer draft B, nor can its ack clear B.
          await store.replaceRejectedEnvelope(item,frame);
          if(gen!==generation)return;
          if(draft.version===item.version) {
            draft={...draft,target_id:frame.target_id,week:frame.week,
              context_revision:frame.context_revision,metadata_confirmed:true};
            target.value=frame.target_id;week.value=String(frame.week);await persist();
          }
        }
      }
      blocked=false;
      if(saved.outbox.length) {
        hint('전송 대기 원문의 대상·주차를 확인했습니다. 새로 작성한 초안은 그대로 보관됩니다.');render();drain();return;
      }
      if(apply) {target.value=state.context.target_id || '';week.value=String(state.context.week);}
      draft={...takeDraft(),context_revision:state.context.revision,metadata_confirmed:true};
      changed();await persist();hint('대상·주차를 확인했습니다. 보내기를 눌러 전송하세요.');
    }catch(e){hint(e.message || '연결 후 전송 여부를 확인해 주세요.');}
  }
  $('context-apply').onclick = () => confirmContext(true);
  $('context-keep').onclick = () => confirmContext(false);
  text.addEventListener('input', changed); target.addEventListener('change', changed); week.addEventListener('change', changed);
  seg.addEventListener('click', e => {const b=e.target.closest('button'); if(b && draft) {draft.source=b.dataset.value; changed();}});
  $('compose-form').addEventListener('submit', async e => {
    e.preventDefault();
    if (!ready || busy || !draft) return;
    if(!text.value.trim()) {hint('의견을 입력해 주세요.'); text.focus(); return;}
    if(!target.value) {hint('대상 학생을 선택해 주세요.'); target.focus(); return;}
    if(Array.from(text.value).length > state.defaults.maxChars) return;
    if(state.context.accepting === false) {hint('현재 제출이 닫혀 있습니다. 원문은 기기에 보관됩니다.'); return;}
    try {
      const saved = await store.readScope(scope);
      if(saved.outbox.length) {hint('이전 원문의 전송 여부를 먼저 확인합니다.'); drain(); return;}
      await persist();
      const current=takeDraft();
      const frame={t:'submit', nonce:store.newVersion(), owner_capability:store.randomToken(), text:current.text,
        source:current.source, week:current.week, target_id:current.target_id,
        context_revision:current.context_revision, metadata_confirmed:current.metadata_confirmed};
      await store.enqueue(current, frame);
      $('c-local').textContent='전송 대기';
      hint(state.connected ? '원문을 전송합니다.' : '연결되면 저장된 원문을 전송합니다.');
      drain();
    } catch(error) {storageError(error);}
  });
  text.addEventListener('keydown', e => {if((e.ctrlKey || e.metaKey) && e.key==='Enter') {e.preventDefault(); $('compose-form').requestSubmit();}});
  function downloadDraft() {
    const link=document.createElement('a'), url=URL.createObjectURL(new Blob([text.value], {type:'text/plain;charset=utf-8'}));
    link.href=url; link.download='feedback-draft.txt';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  $('draft-download').onclick=downloadDraft;
  $('draft-clear').onclick = () => {$('draft-clear-confirm').hidden=false;};
  $('draft-clear-confirm').onclick = async () => {
    if(busy) {hint('전송 결과 확인 후 삭제해 주세요.'); return;}
    try {await store.clearScope(scope); savedVersion=null; fill(defaultDraft()); emitReceipts([]); $('c-local').textContent='이 기기의 기록을 삭제했습니다.'; $('draft-clear-confirm').hidden=true;}
    catch {storageError();}
  };
  document.addEventListener('atlas:hello', initialize);
  document.addEventListener('atlas:context', contextChanged);
  document.addEventListener('atlas:conn', render);
  document.addEventListener('atlas:identity-clear', () => {generation++; clearTimeout(timer); scope=null; draft=null; text.value=''; ready=false;$('c-pending-context').textContent='';$('c-local').textContent='';$('c-summary').textContent='';hint('보내기 전에 대상·주차·출처를 확인하세요.'); emitReceipts([]);});
  setInterval(drain, 4000);
  return {flush:()=>persist().catch(()=>{}), saveBeforeRefresh:persist, downloadDraft};
}
