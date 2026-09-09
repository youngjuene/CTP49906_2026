import {queryState, exportIds, visiblePoints, counts} from './query-state.js';
/* Admin-only tools: search, selection, nearest neighbours, export (PRD 5.6).
 *
 * All four are deliberately admin-only. Not because any of them is unsafe on the
 * participant channel -- selection and search operate on text every participant
 * already has -- but because the participant screen has one job during a
 * presentation, which is to be written on and read from across a room. These are
 * instruments for the person running the session.
 *
 * What makes them safe is unchanged and structural: this file only ever sees the
 * admin channel's serialisation, so `reviewer_name` appears here because the
 * server sent it, not because anything here asked for it.
 */

const CAT = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
             "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];

// A lookup, not a ternary: the previous `student ? 수강생 : 청강생` labelled every
// non-student as 청강생, so a TA appeared as an auditor in the one panel whose
// job is saying who wrote what.
const ROLE_LABEL = { student: "수강생", auditor: "청강생", ta: "조교" };

export function mountAdmin({ state, send, request, atlas, renderStatus, openReader }) {
  const $ = (id) => document.getElementById(id);
  let selection = [];          // point records, in map order
  let picked = null;           // the point whose neighbours are shown

  const isAdmin = () => state.channel === "admin";
  const colourOf = (p) => {
    const i = state.targetIndex.get(p.target_id);
    return i == null || i >= CAT.length ? "var(--other)" : CAT[i];
  };
  const targetName = (id) => state.targetIndex.has(id)
    ? state.targets[state.targetIndex.get(id)].display_name : id;

  /* ---------------- an opinion row, shared by all three lists ------------- */
  function opinionRow(p, { distance = null, query = "" } = {}) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "oprow";
    const text = document.createElement("div");
    text.className = "t";
    if (query) highlight(text, p.text, query); else text.textContent = p.text;
    const meta = document.createElement("div");
    meta.className = "m";
    const sw = document.createElement("span");
    sw.className = "sw"; sw.style.background = colourOf(p);
    const who = document.createElement("span");
    // reviewer_name is present because this is the admin channel. On a
    // participant socket the field does not exist to fall back from.
    who.textContent = `${targetName(p.target_id)} · ${p.source === "ai" ? "AI 생성" : "사람 작성"}`
      + `${p.reviewer_name ? " · " + p.reviewer_name : ""} · ${p.week}주차`;
    meta.append(sw, who);
    if (distance != null) {
      const d = document.createElement("span");
      d.className = "d";
      d.textContent = distance.toFixed(3);
      meta.appendChild(d);
    }
    row.append(text, meta);
    row.addEventListener("click", () => pick(p));
    return row;
  }

  /* Highlight without innerHTML: the text is whatever a participant typed. */
  function highlight(node, text, query) {
    const hay = text.toLowerCase(), needle = query.toLowerCase();
    let at = 0, found = hay.indexOf(needle);
    while (found !== -1 && needle) {
      node.appendChild(document.createTextNode(text.slice(at, found)));
      const mark = document.createElement("mark");
      mark.textContent = text.slice(found, found + needle.length);
      node.appendChild(mark);
      at = found + needle.length;
      found = hay.indexOf(needle, at);
    }
    node.appendChild(document.createTextNode(text.slice(at)));
  }

  /* ---------------- 1. full-text search ---------------------------------- */
  const searchBox = $("corpus-search");

  function runSearch() {
    if (!isAdmin()) return;
    const query = searchBox.value.trim();
    state.query=queryState({...state.query,search:query,
      targetIds:$('filter-target').value?[$('filter-target').value]:[],
      sources:$('filter-source').value?[$('filter-source').value]:[]});
    if (!query) {
      atlas.setMatches(null);
      $("search-count").textContent = "";
      renderStatus();
      return;
    }
    // A substring scan over at most a thousand short strings is instant, and an
    // index would be a dependency bought to replace one loop. Korean makes the
    // simple choice the right one too: word-boundary matching would need a
    // morphological analyser to be better than this, not worse.
    const needle = query.toLowerCase();
    const hits = [...atlas.points.values()]
      .filter((p) => p.text.toLowerCase().includes(needle));
    atlas.setMatches(hits.map((p) => p.id));
    $("search-count").textContent = hits.length
      ? `${hits.length}건 일치` : "일치하는 의견이 없습니다";
    renderStatus();
  }

  searchBox.addEventListener("input", runSearch);
  $('filter-target').onchange=$('filter-source').onchange=runSearch;

  /* ---------------- 2. selection ----------------------------------------- */
  const modeBox = $("select-mode");
  modeBox.addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    atlas.setMode(button.dataset.mode);
    for (const b of modeBox.children)
      b.setAttribute("aria-pressed", String(b === button));
  });

  function setSelection(points) {
    selection = points;
    state.query=queryState({...state.query,selectedIds:points.map(p=>p.id)});
    atlas.setSelection(points.map((p) => p.id));
    const list = $("sel-list");
    list.replaceChildren();
    for (const p of points.slice(0, 200)) list.appendChild(opinionRow(p));
    $("sel-count").textContent = points.length
      ? `${points.length}건 선택됨` + (points.length > 200 ? " (200건까지 표시)" : "")
      : "지도를 끌어 여러 의견을 한 번에 고릅니다.";
    renderStatus();
  }

  $("btn-clear-sel").addEventListener("click", () => setSelection([]));

  /* ---------------- 3. nearest neighbours -------------------------------- */
  function pick(p) {
    if (!isAdmin() || !p) return;
    picked = p;
    $("pane-neighbors").hidden = false;
    $("nb-of").textContent = `기준: ${p.text.slice(0, 40)}${p.text.length > 40 ? "…" : ""}`;
    $("nb-list").replaceChildren();
    send({ t: "neighbors", id: p.id, k: 8 });
  }

  document.addEventListener("atlas:neighbors", (e) => {
    const { id, ids, distances, ready } = e.detail;
    if (!picked || picked.id !== id) return;
    const list = $("nb-list");
    list.replaceChildren();
    ids.forEach((oid, i) => {
      const p = atlas.points.get(oid);
      if (p) list.appendChild(opinionRow(p, { distance: distances[i] }));
    });
    if (!list.childElementCount) {
      const none = document.createElement("p");
      none.className = "hint";
      // "Still being computed" and "genuinely has no neighbours" are both an
      // empty list and mean opposite things to the person reading the panel.
      none.textContent = ready === false
        ? "배치를 계산하는 중입니다. 잠시 후 다시 눌러 주세요."
        : "가까운 의견이 없습니다.";
      list.appendChild(none);
    }
  });

  /* ---------------- 4. export -------------------------------------------- */
  let preview=null;
  function renderPreview() {
    const scope=$('export-scope').value, format=$('export-format').value;
    const ids=exportIds(scope,atlas.points,state.query);
    const rows=format==='originals'?new Set(ids.map(id=>atlas.points.get(id).submission_id || id)).size:ids.length;
    preview={scope,format,ids,expected_data_rev:state.rev,rows};
    $('export-preview').textContent=`${$('export-scope').selectedOptions[0].textContent} · ${rows}행 · 데이터 ${state.rev}`;
    $('btn-export').disabled=!rows;
  }
  document.addEventListener('atlas:query',renderPreview);
  $('export-scope').onchange=$('export-format').onchange=renderPreview;
  $('btn-export').addEventListener('click',async()=>{
    if(!isAdmin() || !preview)return;
    const frozen={...preview,ids:[...preview.ids]},note=$('export-note');
    note.textContent='내보내는 중…';
    try {
      const res=await fetch('/api/export.csv',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({
        code:state.adminCode, scope:frozen.scope,format:frozen.format,expected_data_rev:frozen.expected_data_rev,
        ...(frozen.scope==='all'?{}:{ids:frozen.ids})})});
      if(res.status===409){send({t:'resync',have_rev:state.rev});renderPreview();note.textContent='데이터가 바뀌었습니다. 갱신된 범위와 행 수를 확인하고 다시 눌러 주세요.';return;}
      if(!res.ok){note.textContent='내보내기에 실패했습니다.';return;}
      const url=URL.createObjectURL(await res.blob()),a=document.createElement('a');
      a.href=url;a.download=`feedback-atlas-${frozen.format}.csv`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
      note.textContent=`${frozen.rows}행을 내보냈습니다.`;
    }catch{note.textContent='연결을 확인하고 다시 눌러 주세요.';}
  });

  function renderAdminCounts() {
    if(!isAdmin() || !state.adminCounts)return;
    const c=state.adminCounts;
    $('admin-counts').textContent=`전체 의견 ${c.units}개 · 원문 ${c.submissions}건 · 제출자 ${c.submitters}명`;
  }
  function renderProcessing() {
    const c=state.processing || {},d=state.processingDiagnostics;
    if(d) $('processing-diagnostics').textContent=`${!(c.queued || c.splitting || c.embedding)?'대기 원문 없음':`가장 오래 기다린 원문 ${Math.round(d.oldest_pending_seconds)}초`} · 분할 실패 ${d.failed_by_stage?.splitting || 0}건 · 지도 반영 실패 ${d.failed_by_stage?.embedding || 0}건`;
    $('processing-counts').textContent=`대기 ${c.queued || 0} · 처리 중 ${(c.splitting || 0)+(c.embedding || 0)} · 실패 ${c.failed || 0} · 반영 완료 ${c.ready || 0} · 철회 ${c.withdrawn || 0}`;
  }
  function renderContext() {
    if(!isAdmin())return;
    $('class-target').replaceChildren(new Option('발표 대상 없음',''),...state.targets.map(t=>new Option(t.display_name,t.id)));
    $('class-target').value=state.context.target_id || '';
    $('class-week').value=String(state.context.week);$('class-accepting').checked=state.context.accepting;
    renderProcessing();
  }
  $('class-save').onclick=async()=>{
    try {await request({t:'class_context_update',expected_revision:state.context.revision,target_id:$('class-target').value || null,
      week:Number($('class-week').value),accepting:$('class-accepting').checked});$('class-status').textContent='수업 정보가 적용되었습니다.';}
    catch(e){$('class-status').textContent=`${e.message} 최신 수업 정보를 확인해 주세요.`;}
  };
  document.addEventListener('atlas:context',renderContext);
  document.addEventListener('atlas:processing',renderProcessing);

  /* ---------------- reviewer list (unchanged behaviour) ------------------- */
  const revSearch = $("rev-search");

  function counts() {
    const out = new Map();
    for (const p of atlas.points.values()) {
      if (!p.reviewer_id) continue;
      out.set(p.reviewer_id, (out.get(p.reviewer_id) || 0) + 1);
    }
    return out;
  }

  function renderReviewers() {
    if (!isAdmin()) return;
    const query = revSearch.value.trim().toLowerCase();
    const written = counts();
    const list = $("rev-list");
    list.replaceChildren();
    const rows = state.roster
      .filter((r) => !query || `${r.display_name} ${r.id}`.toLowerCase().includes(query))
      .map((r) => ({ ...r, n: written.get(r.id) || 0 }))
      .sort((a, b) => b.n - a.n || a.display_name.localeCompare(b.display_name, "ko"));

    for (const r of rows) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "revrow";
      row.setAttribute("aria-selected", String(state.focusReviewer === r.id));
      const name = document.createElement("span");
      name.className = "n"; name.textContent = r.display_name;
      const role = document.createElement("span");
      role.className = "rolechip";
      role.textContent = ROLE_LABEL[r.role] || r.role;
      const n = document.createElement("span");
      n.className = "c"; n.textContent = String(r.n);
      if (r.n === 0) row.style.opacity = "0.55";
      row.append(name, role, n);
      row.addEventListener("click", () => {
        state.focusReviewer = state.focusReviewer === r.id ? null : r.id;
        state.query=queryState({...state.query,emphasis:{...state.query.emphasis,reviewerId:state.focusReviewer}});
        const status = $("status-msg");
        status.textContent = !state.focusReviewer ? ""
          : r.n ? `${r.display_name}의 의견만 강조 중`
                : `${r.display_name}이(가) 남긴 의견이 없습니다`;
        renderReviewers(); atlas.render(); renderStatus();
      });
      list.appendChild(row);
    }
    if (!rows.length) {
      const none = document.createElement("p");
      none.className = "hint";
      none.textContent = "일치하는 작성자가 없습니다.";
      list.appendChild(none);
    }
  }

  revSearch.addEventListener("input", renderReviewers);
  document.addEventListener("atlas:hello", () => {
    if (!isAdmin()) return;
    atlas.setMode("pan");
    const oldTarget=$('filter-target').value;
    $('filter-target').replaceChildren(new Option('전체 대상',''),...state.targets.map(t=>new Option(t.display_name,t.id)));
    $('filter-target').value=oldTarget;
    renderContext();renderReviewers();renderPreview();renderAdminCounts();
  });
  document.addEventListener("atlas:data", () => {
    selection=selection.filter(p=>atlas.points.has(p.id));
    if(isAdmin())setSelection(selection.map(p=>atlas.points.get(p.id)));
    if(picked && !atlas.points.has(picked.id)) {picked=null;$('nb-list').replaceChildren();$('pane-neighbors').hidden=true;}
    renderReviewers(); runSearch();renderAdminCounts();
  });

  // Wire the map's callbacks. Both are no-ops on the participant channel.
  atlas.onSelect = (points) => { if (isAdmin()) setSelection(points); };
  atlas.onPick = point => {
    if(!point)return;
    if(isAdmin())pick(point); else openReader(point,document.getElementById('map'));
  };
}
