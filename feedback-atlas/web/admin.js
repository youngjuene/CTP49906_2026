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

export function mountAdmin({ state, send, atlas, renderStatus }) {
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
    who.textContent = `${targetName(p.target_id)} · ${p.source === "ai" ? "AI" : "사람"}`
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
    atlas.setSelection(points.map((p) => p.id));
    const list = $("sel-list");
    list.replaceChildren();
    for (const p of points.slice(0, 200)) list.appendChild(opinionRow(p));
    $("sel-count").textContent = points.length
      ? `${points.length}건 선택됨` + (points.length > 200 ? " (200건까지 표시)" : "")
      : "지도를 끌어 여러 의견을 한 번에 고릅니다.";
    $("export-note").textContent = "";
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
  $("btn-export").addEventListener("click", async () => {
    if (!isAdmin()) return;
    const note = $("export-note");
    note.textContent = "내보내는 중…";
    try {
      // POST with the code in the body, never a query string -- a URL carrying
      // the admin code would sit in tunnel logs and browser history.
      const res = await fetch("/api/export.csv", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          code: state.adminCode,
          ids: selection.length ? selection.map((p) => p.id) : undefined,
        }),
      });
      if (!res.ok) { note.textContent = "내보내기에 실패했습니다."; return; }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `feedback-atlas-${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      note.textContent = selection.length
        ? `선택한 ${selection.length}건을 내보냈습니다.`
        : "전체를 내보냈습니다.";
    } catch {
      note.textContent = "내보내기에 실패했습니다.";
    }
  });

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
      role.textContent = r.role === "student" ? "수강생" : "청강생";
      const n = document.createElement("span");
      n.className = "c"; n.textContent = String(r.n);
      if (r.n === 0) row.style.opacity = "0.55";
      row.append(name, role, n);
      row.addEventListener("click", () => {
        state.focusReviewer = state.focusReviewer === r.id ? null : r.id;
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
    renderReviewers();
  });
  document.addEventListener("atlas:data", () => { renderReviewers(); runSearch(); });

  // Wire the map's callbacks. Both are no-ops on the participant channel.
  atlas.onSelect = (points) => { if (isAdmin()) setSelection(points); };
  atlas.onPick = (point) => { if (isAdmin() && point) pick(point); };
}
