/* Owner-only capabilities remain local; originals load only when requested. */
export function mountSubmissionStatus({request, state, openReader}) {
  const box=document.getElementById('submission-status');
  let receipts=[], details=new Map(), generation=0;
  const labels={queued:'처리 대기', splitting:'의견을 나누는 중', embedding:'지도에 반영 중', ready:'반영됨', failed:'처리 실패', withdrawn:'철회됨'};
  function render() {
    box.replaceChildren();
    for(const receipt of receipts.slice().reverse()) {
      const detail=details.get(receipt.submission_id) || receipt;
      const card=document.createElement('article'); card.className='submission-card'; card.dataset.submissionId=receipt.submission_id;
      const title=document.createElement('p');
      title.textContent=`원문 저장됨 · ${labels[detail.state] || '상태 확인 중'}`;
      if(detail.state==='ready') title.textContent += ` · 의견 ${detail.units?.length || 0}개로 나누어 반영했습니다`;
      const inspect=document.createElement('button'); inspect.className='btn'; inspect.textContent='확인·수정';
      inspect.onclick=()=>openReader({submission_id:receipt.submission_id, owner_capability:receipt.owner_capability}, inspect);
      card.append(title);
      if(detail.raw_text) {const excerpt=document.createElement('p');excerpt.className='submission-excerpt';excerpt.textContent=detail.raw_text.slice(0,100);card.append(excerpt);}
      card.append(inspect);
      if(detail.state==='failed') {
        const retry=document.createElement('button'); retry.className='btn'; retry.textContent='처리 다시 시도';
        retry.onclick=async()=>{try{ const result=await request({t:'submission_retry',submission_id:receipt.submission_id,owner_capability:receipt.owner_capability}); details.set(result.submission_id,result);render();}catch(e){title.textContent=`원문 저장됨 · ${e.message}`;}};
        card.append(retry);
      }
      box.append(card);
    }
  }
  async function refresh() {
    const gen=generation;
    if(!state.connected) return;
    for(const receipt of receipts) {
      try {
        const detail=await request({t:'submission_status', submission_id:receipt.submission_id, owner_capability:receipt.owner_capability});
        if(gen !== generation) return;
        details.set(detail.submission_id, detail);
      } catch {/* Retain the saved receipt through a temporary disconnection. */}
    }
    if(gen===generation) render();
  }
  document.addEventListener('atlas:receipts',e=>{receipts=e.detail; generation++; if(!receipts.length)details.clear(); render(); refresh();});
  document.addEventListener('atlas:submission_detail',e=>{
    if(receipts.some(r=>r.submission_id===e.detail.submission_id)) {details.set(e.detail.submission_id,e.detail);render();}
  });
  document.addEventListener('atlas:conn',()=>{if(state.connected)refresh();});
  setInterval(()=>{if(receipts.some(r=>!['ready','withdrawn'].includes(details.get(r.submission_id)?.state)))refresh();},7000);
}
