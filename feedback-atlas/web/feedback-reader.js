import {newVersion} from './draft-store.js';
const sourceName = source => source==='ai' ? 'AI 생성' : '사람 작성';
export function mountReader({state, request}) {
  const $=id=>document.getElementById(id), dialog=$('feedback-reader');
  let detail, manifest=[], owner=null, returnFocus=null, dirty=false, pendingAction=null, waiting=false, generation=0;
  function lockControls() {
    const locked=waiting || !!pendingAction;
    for(const node of dialog.querySelectorAll('.unit-source,.unit-split,.unit-merge,#reader-target,#reader-week,#reader-save,#reader-withdraw,#reader-confirm-withdraw')) node.disabled=locked;
    $('reader-resolve').hidden=!pendingAction;
    $('reader-resolve').disabled=waiting;
    $('reader-resolve').textContent=pendingAction?.t==='submission_withdraw'?'이전 철회 결과 다시 확인':'이전 수정 저장 결과 다시 확인';
    if(pendingAction)$('reader-confirm-withdraw').hidden=true;
  }
  function history() {
    $('reader-history').textContent=(detail.history || []).map(h=>{
      let units=[];try{units=JSON.parse(h.manifest || '[]');}catch{/* Historical data stays readable without exposing storage JSON. */}
      const name=state.targets.find(t=>t.id===h.target_id)?.display_name || h.target_id || '대상 정보 없음';
      const ai=units.filter(u=>u.source==='ai').length,human=units.filter(u=>u.source==='human').length;
      return `수정 ${h.revision} · ${name} · ${h.week || '?'}주차 · 의견 ${units.length}개\nAI 생성 ${ai}개 · 사람 작성 ${human}개${h.timestamp?' · '+new Date(h.timestamp).toLocaleString('ko-KR'):''}`;
    }).join('\n\n');
  }
  function renderUnits() {
    const box=$('reader-units'); box.replaceChildren();
    const chars=Array.from(detail.raw_text);
    manifest.forEach((unit,index)=>{
      const row=document.createElement('section'); row.className='reader-unit';
      const heading=document.createElement('p'); heading.textContent=`의견 ${index+1} · ${sourceName(unit.source)} · 원문 ${unit.start}–${unit.end}자`;
      const quote=document.createElement(owner?'textarea':'pre');
      if(owner){quote.className='unit-text';quote.readOnly=true;quote.value=chars.slice(unit.start,unit.end).join('');quote.setSelectionRange(0,0);quote.setAttribute('aria-label',`의견 ${index+1}: 나눌 위치를 클릭하세요`);quote.addEventListener('pointerup',()=>{dirty=true;});quote.addEventListener('keydown',event=>{
        dirty=true;
        // Chromium scrolls a readonly textarea on unmodified arrow keys.
        // Move its selection explicitly so keyboard users can place a boundary.
        if(!event.shiftKey && !event.altKey && !event.ctrlKey && !event.metaKey && ['ArrowLeft','ArrowRight'].includes(event.key)) {
          event.preventDefault();
          if(!Intl.Segmenter) {$('reader-message').textContent='이 브라우저에서는 마우스나 터치로 나눌 위치를 선택하세요.';return;}
          const stops=[...new Intl.Segmenter('ko',{granularity:'grapheme'}).segment(quote.value)].map(part=>part.index);
          stops.push(quote.value.length);
          const caret=event.key==='ArrowRight' ? stops.find(at=>at>quote.selectionStart) ?? quote.value.length
            : stops.filter(at=>at<quote.selectionStart).at(-1) ?? 0;
          quote.setSelectionRange(caret,caret);
        }else if(!event.shiftKey && ['Home','End'].includes(event.key)) {
          event.preventDefault();let caret;
          if(event.key==='Home')caret=event.ctrlKey || event.metaKey?0:quote.value.lastIndexOf('\n',quote.selectionStart-1)+1;
          else {caret=event.ctrlKey || event.metaKey?quote.value.length:quote.value.indexOf('\n',quote.selectionStart);if(caret<0)caret=quote.value.length;}
          quote.setSelectionRange(caret,caret);
        }
      });}
      else quote.textContent=chars.slice(unit.start,unit.end).join('');
      row.append(heading,quote);
      if(owner && detail.state!=='withdrawn') {
        const label=document.createElement('label'); label.textContent='피드백 출처 ';
        const select=document.createElement('select'); select.className='unit-source';
        select.append(new Option('AI 생성','ai'),new Option('사람 작성','human')); select.value=unit.source;
        select.onchange=()=>{unit.source=select.value;dirty=true;}; label.append(select); row.append(label);
        const splitLabel=document.createElement('p');splitLabel.className='hint';splitLabel.textContent='위 의견에서 나눌 곳을 클릭하거나 방향키로 이동한 뒤 나누기를 누르세요.';
        const preview=document.createElement('p');preview.className='hint split-preview';preview.textContent='나눌 곳을 먼저 클릭하세요.';
        const showBoundary=()=>{const at=quote.selectionStart;preview.textContent=at>0 && at<quote.value.length?
          `나눌 위치: “${quote.value.slice(0,at).slice(-24)}” │ “${quote.value.slice(at,at+24)}”`:'나눌 곳을 먼저 클릭하세요.';};
        quote.addEventListener('select',showBoundary);quote.addEventListener('pointerup',showBoundary);quote.addEventListener('keyup',showBoundary);
        const split=document.createElement('button');split.className='btn unit-split';split.textContent='여기서 나누기';
        split.onclick=()=>{
          const offset=quote.selectionStart;
          const wanted=Array.from(quote.value.slice(0,offset)).length;
          let local=0, seen=0;
          while(seen<wanted && unit.start+local<unit.end) {local += chars[unit.start+local]==='\r' && chars[unit.start+local+1]==='\n'?2:1;seen++;}
          const at=unit.start+local;
          if(!offset || !Number.isInteger(at) || at<=unit.start || at>=unit.end || !chars.slice(unit.start,at).join('').trim() || !chars.slice(at,unit.end).join('').trim()) {$('reader-message').textContent='나눌 곳을 먼저 클릭하세요. 양쪽에 의견이 남는 위치를 선택해 주세요.';return;}
          if(manifest.length>=64){$('reader-message').textContent='최대 64개 의견까지 나눌 수 있습니다.';return;}
          manifest.splice(index,1,{...unit,end:at},{...unit,start:at});dirty=true;renderUnits();
        };
        row.append(splitLabel,preview,split);
        if(index<manifest.length-1) {
          const merge=document.createElement('button');merge.className='btn unit-merge';merge.textContent='다음 의견과 합치기';
          merge.onclick=()=>{manifest.splice(index,2,{...unit,end:manifest[index+1].end});dirty=true;renderUnits();$('reader-message').textContent='합친 의견의 출처를 확인하세요.';};row.append(merge);
        }
      }
      box.append(row);
    });
  }
  function render() {
    $('reader-original').textContent=detail.raw_text;
    $('reader-title').textContent=detail.state==='withdrawn'?'철회됨':'원문과 의견 확인';
    $('reader-target').replaceChildren(...state.targets.map(t=>new Option(t.display_name,t.id)));
    $('reader-target').value=detail.target_id;
    $('reader-week').value=String(detail.week);
    $('reader-metadata').hidden=!owner;
    $('reader-save').hidden=!owner || detail.state==='withdrawn';
    $('reader-withdraw').hidden=!owner || detail.state==='withdrawn';
    $('reader-confirm-withdraw').hidden=true;
    $('reader-history-section').hidden=!owner;
    history();renderUnits();lockControls();
  }
  async function load() {
    return request({t:owner?'submission_status':'submission_original',submission_id:detail.submission_id,...(owner?{owner_capability:owner.owner_capability}:{})});
  }
  async function open(reference, trigger=document.activeElement) {
    const gen=++generation;
    owner=reference.owner_capability || state.channel==='admin' ? reference : null;
    detail={submission_id:reference.submission_id};returnFocus=trigger;dirty=false;pendingAction=null;waiting=false;
    $('reader-message').textContent='원문을 불러오는 중…';$('reader-history').textContent='';$('reader-history-section').hidden=true;$('reader-resolve').hidden=true;$('reader-confirm-withdraw').hidden=true;
    $('reader-units').replaceChildren();$('reader-original').textContent='';
    $('reader-save').hidden=true;$('reader-withdraw').hidden=true;$('reader-metadata').hidden=true;
    if(!dialog.open)dialog.showModal();
    try {
      const result=await load(); if(gen!==generation)return;
      detail=result;manifest=detail.units.map(u=>({start:u.start,end:u.end,source:u.source}));
      $('reader-message').textContent='';render();
    }catch(e){if(gen===generation)$('reader-message').textContent=e.message || '원문을 불러오지 못했습니다.';}
  }
  async function act(kind) {
    if(waiting || !owner)return;
    if(pendingAction && pendingAction.t!==kind) {
      $('reader-message').textContent='이전 전송의 결과를 먼저 다시 확인해 주세요. 다른 작업은 그 뒤에 할 수 있습니다.';return;
    }
    if(!pendingAction) pendingAction={t:kind, submission_id:detail.submission_id, owner_capability:owner.owner_capability,
      expected_revision:detail.revision, action_nonce:newVersion(), ...(kind==='submission_correct'?{
        units:manifest.map(u=>({...u})),target_id:$('reader-target').value,week:Number($('reader-week').value)}:{})};
    const gen=generation, action=pendingAction;
    const current=()=>gen===generation && detail?.submission_id===action.submission_id && pendingAction===action;
    waiting=true;lockControls();
    try {
      const result=await request(action);
      if(!current())return;
      detail=result;pendingAction=null;dirty=false;
      manifest=detail.units.map(u=>({start:u.start,end:u.end,source:u.source}));render();
      $('reader-message').textContent=detail.state==='withdrawn'?'철회됨':'수정 내용이 저장되었습니다. 지도 반영 상태를 확인하세요.';
    }catch(e){
      if(!current())return;
      if(e.code && !['OFFLINE','TIMEOUT'].includes(e.code)) {
        pendingAction=null;
        if(['REVISION_CONFLICT','STALE_REVISION'].includes(e.code)) {
          try {
            const latest=await load();if(gen!==generation)return;
            detail=latest;manifest=detail.units.map(u=>({start:u.start,end:u.end,source:u.source}));dirty=false;render();
          }catch{/* Keep the original and the error visible if refresh is offline. */}
        }
      }
      if(gen===generation)$('reader-message').textContent=pendingAction?
        `${e.message} 이전 전송 결과 다시 확인을 눌러 저장 여부를 확인하세요.`:
        `${e.message} 최신 내용을 확인하고 다시 눌러 주세요.`;
    }finally{if(gen===generation){waiting=false;lockControls();}}
  }
  $('reader-resolve').onclick=()=>{if(pendingAction)act(pendingAction.t);};
  $('reader-save').onclick=()=>act('submission_correct');
  $('reader-withdraw').onclick=()=>{$('reader-confirm-withdraw').hidden=false;};
  $('reader-confirm-withdraw').onclick=()=>act('submission_withdraw');
  $('reader-close').onclick=()=>dialog.close();
  $('reader-target').onchange=$('reader-week').onchange=()=>{dirty=true;};
  dialog.addEventListener('close',()=>{
    generation++;
    if(returnFocus?.isConnected)returnFocus.focus();
    else if(returnFocus?.dataset?.unitId) [...document.querySelectorAll('#visible-results button')].find(b=>b.dataset.unitId===returnFocus.dataset.unitId)?.focus();
    else [...document.querySelectorAll('.submission-card')].find(card=>card.dataset.submissionId===detail?.submission_id)?.querySelector('button')?.focus();
  });
  document.addEventListener('atlas:submission_detail',e=>{
    if(dialog.open && detail?.submission_id===e.detail.submission_id && !dirty && !waiting && owner && (detail.revision!==e.detail.revision || detail.state!==e.detail.state || detail.units.length!==e.detail.units.length)) {
      detail=e.detail;manifest=detail.units.map(u=>({start:u.start,end:u.end,source:u.source}));render();
    }
  });
  document.addEventListener('atlas:identity-clear',()=>{dialog.close();detail=null;owner=null;$('reader-original').textContent='';$('reader-history').textContent='';$('reader-units').replaceChildren();});
  return open;
}
