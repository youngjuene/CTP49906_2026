/* Entry point: connection, state, and the three views.
 *
 * The connection is the interesting part. WebSocket has no EventSource-style
 * auto-reconnect, and Cloudflare quick tunnels drop an idle socket at roughly
 * 100 seconds -- which lands in the quiet middle of a presentation. So: a backoff
 * loop, a reply to the server's heartbeat, and a resync whenever the revision
 * we hold is not the one a delta expects.
 */
import { Atlas, CATEGORY10 } from "./atlas.js";
import { mountCompose } from "./compose.js";
import { mountAdmin } from "./admin.js";
import * as viewer from "./viewer.js";

const PROTOCOL = 1;
const OTHER = "var(--other)";

const state = {
  channel: "participant",
  identity: null, adminCode: null,
  rev: 0, layoutRev: 0,
  targets: [], targetIndex: new Map(),
  roster: [],
  weeks: new Set([1, 2, 3, 4]),
  focusTarget: null, focusReviewer: null,
  defaults: { source: "ai", week: 1, maxChars: 1000 },
  connected: false,
};

const $ = (id) => document.getElementById(id);
const views = { gate: $("view-gate"), admin: $("view-admin-gate"), main: $("view-main") };

function show(name) {
  for (const [k, node] of Object.entries(views)) node.hidden = k !== name;
}

/* ---------------- theme ---------------- */
const THEME_KEY = "atlas.theme";
try {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved) document.documentElement.dataset.theme = saved;
} catch { /* private mode; the OS preference still applies */ }

$("btn-theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme
    ? root.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem(THEME_KEY, root.dataset.theme); } catch { /* ignore */ }
  // The viewer paints its own chrome and is told the scheme rather than reading
  // it: it renders into a canvas, so a CSS variable change never reaches it.
  viewer.setColorScheme(colorScheme());
});

function colorScheme() {
  const root = document.documentElement;
  if (root.dataset.theme) return root.dataset.theme;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/* ---------------- the map ---------------- */
const atlas = new Atlas($("map"), { onHover: showTip });
atlas.colorOf = (p) => {
  const i = state.targetIndex.get(p.target_id);
  return i == null || i >= CATEGORY10.length ? OTHER : CATEGORY10[i];
};
atlas.labelOf = (id) => state.targetIndex.has(id)
  ? state.targets[state.targetIndex.get(id)].display_name : id;
atlas.visible = (p) => state.weeks.has(Number(p.week));
atlas.dimmed = (p) =>
  (state.focusTarget != null && p.target_id !== state.focusTarget) ||
  (state.focusReviewer != null && p.reviewer_id !== state.focusReviewer) ||
  (atlas.matches != null && !atlas.matches.has(p.id));

/* ---------------- tooltip ---------------- */
let tipEl = null;
function showTip(hit) {
  tipEl?.remove(); tipEl = null;
  if (!hit) return;
  const p = hit.point;
  const card = $("map-card");
  tipEl = document.createElement("div");
  tipEl.className = "tip";
  const colour = atlas.colorOf(p);
  tipEl.innerHTML =
    `<div><div class="flabel" style="margin-bottom:4px">의견</div>
       <div class="text"></div></div>
     <div class="meta">
       <span class="badge"><span class="sw" style="background:${colour}"></span>
         <span class="tgt"></span></span>
       <span class="badge">${Number(p.week)}주차</span>
       <span class="badge">${p.source === "ai" ? "AI" : "사람"}</span>
       ${p.reviewer_name ? '<span class="badge rev"></span>' : ""}
     </div>`;
  // textContent, never innerHTML, for anything a participant typed.
  tipEl.querySelector(".text").textContent = p.text;
  tipEl.querySelector(".tgt").textContent = atlas.labelOf(p.target_id);
  if (p.reviewer_name) tipEl.querySelector(".rev").textContent = p.reviewer_name;

  card.appendChild(tipEl);
  const box = card.getBoundingClientRect(), t = tipEl.getBoundingClientRect();
  let left = hit.cx - t.width / 2;
  let top = hit.cy - t.height - 8;
  if (top < 2) top = hit.cy + 12;                       // flip below near the top
  left = Math.max(2, Math.min(left, box.width - t.width - 2));
  tipEl.style.left = `${left}px`;
  tipEl.style.top = `${top}px`;
}

/* ---------------- chrome ---------------- */
function renderStatus() {
  const total = atlas.points.size;
  let shown = 0;
  for (const p of atlas.points.values()) if (atlas.visible(p)) shown++;
  $("status-count").textContent = `${shown}개 표시 / 전체 ${total}개`;
  $("empty").hidden = total > 0;
  if (total && !shown) $("status-msg").textContent = "주차가 모두 꺼져 있습니다";
  const bar = atlas.scaleBar();
  $("scale").innerHTML =
    `<svg viewBox="0 0 ${bar.px} 9" width="${bar.px}" aria-hidden="true">
       <path d="M1 1v7M${bar.px - 1} 1v7M1 4.5h${bar.px - 2}"/></svg>${bar.label}`;
}

function renderLegend() {
  const box = $("legend");
  box.replaceChildren();
  const h = document.createElement("h4");
  h.textContent = "대상 학생";
  box.appendChild(h);
  state.targets.forEach((t, i) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "row" + (state.focusTarget && state.focusTarget !== t.id ? " off" : "");
    const colour = i < CATEGORY10.length ? CATEGORY10[i] : OTHER;
    row.innerHTML = `<span class="sw" style="background:${colour}"></span><span></span>`;
    row.lastElementChild.textContent = t.display_name;
    row.addEventListener("click", () => {
      state.focusTarget = state.focusTarget === t.id ? null : t.id;
      $("status-msg").textContent = state.focusTarget
        ? `${t.display_name} 강조 중` : "";
      renderLegend(); atlas.render(); renderStatus();
    });
    box.appendChild(row);
  });
  const sep = document.createElement("div"); sep.className = "sep"; box.appendChild(sep);
  const h2 = document.createElement("h4"); h2.textContent = "구분"; box.appendChild(h2);
  box.insertAdjacentHTML("beforeend",
    `<div class="shape"><svg viewBox="-6 -6 12 12"><circle r="4.6" fill="${OTHER}"/></svg>사람</div>
     <div class="shape"><svg viewBox="-6 -6 12 12"><path d="M0-5.2 5.2 0 0 5.2-5.2 0Z" fill="${OTHER}"/></svg>AI</div>`);
}

$("weeks").addEventListener("change", (e) => {
  if (e.target.type !== "checkbox") return;
  const week = Number(e.target.value);
  if (e.target.checked) state.weeks.add(week); else state.weeks.delete(week);
  showTip(null);
  // Coordinates are untouched: filtering only changes opacity, so the map
  // demonstrably does not reflow as weeks are toggled (PRD 5.5).
  atlas.render(); renderStatus();
});

$("btn-fit").addEventListener("click", () => { atlas.fit(); renderStatus(); });
$("tab-compose").addEventListener("click", (e) => {
  if (state.channel === "admin") return;   // there is nothing to show
  toggleSide(e.currentTarget, "compose");
});
$("tab-legend").addEventListener("click", (e) => {
  const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
  e.currentTarget.setAttribute("aria-pressed", String(on));
  $("legend").hidden = !on;
});
function toggleSide(button, id) {
  const on = button.getAttribute("aria-pressed") !== "true";
  button.setAttribute("aria-pressed", String(on));
  $(id).hidden = !on;
}

/* ---------------- Embedding Atlas viewer ----------------
 *
 * Two front ends over one corpus, and the choice between them is a capability
 * question rather than a preference. Apple's renderer has been WebGPU-only since
 * 0.24.0, which dropped the WebGL2 fallback -- so on a machine without it the
 * embedding view is a blank canvas rather than an error. That is unacceptable in
 * a room of assorted laptops, so the hand-written scatter stays as the fallback
 * and the button disables itself where the viewer cannot draw.
 *
 * The default is also deliberate: the viewer is a dashboard, and a phone held by
 * somebody who is there to write a sentence is not where it earns its space. Wide
 * screen and a working adapter, or it stays off until asked for.
 */
const viewerBtn = $("tab-viewer");
let viewerUsable = null;         // null until probed; probing costs an adapter request
let viewerOn = false;
// null until the reader says so, then their answer. Kept separate from viewerOn
// because the two mean different things: viewerOn is what is on screen right now,
// this is what the reader asked for. Without the distinction, auto-enable runs
// again on every reconnect and reopens a viewer somebody closed on purpose -- and
// a quick tunnel drops an idle socket about every hundred seconds, so "every
// reconnect" means several times per presentation.
let viewerChoice = null;

async function viewerIsUsable() {
  if (viewerUsable === null) viewerUsable = await viewer.rendererAvailable();
  return viewerUsable;
}

function viewerNote(text) {
  const note = $("viewer-note");
  note.hidden = !text;
  note.textContent = text || "";
}

/* Say once that this browser cannot open the viewer, and stop offering it.
 *
 * The map is never touched here. An earlier version hid it and put the
 * explanation in the viewer's own panel, so a reader on a laptop without WebGPU
 * pressed the button and lost the map behind a note telling them the map still
 * worked. The message belongs in the status bar, which sits under the map that
 * is still there. */
function markViewerUnavailable() {
  viewerBtn.disabled = true;
  viewerBtn.setAttribute("aria-pressed", "false");
  viewerBtn.title = "분석 보기: 이 브라우저에서는 열 수 없습니다";
  $("status-msg").textContent =
    "이 브라우저는 분석 보기를 지원하지 않습니다 (WebGPU 필요). 지도는 그대로 쓸 수 있습니다.";
}

async function setViewer(on) {
  if (on && !(await viewerIsUsable())) {
    markViewerUnavailable();
    return;                       // the map stays exactly where it was
  }
  if (on && viewerChoice === false) return; // a later click cancelled this probe
  viewerOn = on;
  viewerBtn.setAttribute("aria-pressed", String(on));
  $("viewer-card").hidden = !on;
  $("map-card").hidden = on;
  if (!on) return;
  viewerNote("");
  try {
    await viewer.mount($("viewer-mount"), {
      // A function, not a value: this page can become an admin connection after
      // the viewer was mounted, and a captured null would keep the admin looking
      // at the participant relation.
      getCode: () => (state.channel === "admin" ? state.adminCode : null),
      colorScheme: colorScheme(),
    });
    viewer.refresh();
  } catch (err) {
    viewerNote(`분석 보기를 불러오지 못했습니다. ${String(err.message || err)}`);
  }
}

viewerBtn.addEventListener("click", () => {
  // Pressing the button is the reader stating a preference, and it outranks the
  // default from here on.
  viewerChoice = !(viewerChoice ?? viewerOn);
  setViewer(viewerChoice);
});

async function autoEnableViewer() {
  // Runs on every handshake, including the ones after a dropped tunnel.
  if (viewerChoice !== null) return;      // the reader has already decided
  if (viewerOn) return;
  // Probed regardless of screen width, because the answer also decides whether
  // the button is worth offering at all -- and a narrow screen with a capable
  // browser should still be able to open it by hand.
  if (!(await viewerIsUsable())) {
    markViewerUnavailable();
    return;
  }
  if (viewerChoice !== null || viewerOn) return; // a click can arrive while probing
  if (window.innerWidth >= 1024) setViewer(true);
}

function setConn(kind, text) {
  const node = $("conn");
  node.className = "conn" + (kind === "ok" ? "" : ` ${kind}`);
  $("conn-text").textContent = text;
  state.connected = kind === "ok";
  document.dispatchEvent(new CustomEvent("atlas:conn", { detail: state.connected }));
}

/* ---------------- connection ---------------- */
let socket = null, backoff = 500, closing = false, retryTimer = null;
let lastFrameAt = 0, watchdog = null;

/* A dropped connection does not always announce itself.
 *
 * A close event is the polite case. A tunnel that stops forwarding, a phone that
 * changes network, a laptop coming back from sleep -- these leave the socket in
 * readyState OPEN with nothing ever arriving again, and the client would sit
 * there showing a stale map and an enabled submit button for the rest of the
 * session.
 *
 * The server already sends a heartbeat every 25s. This is the half that makes it
 * useful: if two of them fail to arrive, treat the socket as gone and reconnect.
 */
const LIVENESS_TIMEOUT_MS = 60000;

function startWatchdog() {
  clearInterval(watchdog);
  watchdog = setInterval(() => {
    if (closing || !socket || socket.readyState !== WebSocket.OPEN) return;
    if (Date.now() - lastFrameAt < LIVENESS_TIMEOUT_MS) return;
    setConn("warn", "재연결 중…");
    try { socket.close(); } catch { /* already gone */ }   // onclose schedules the retry
  }, 10000);
}

function wsUrl(path) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}

function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message));
    return true;
  }
  return false;
}

function connect() {
  closing = false;
  clearTimeout(retryTimer);
  retryTimer = null;

  const admin = state.channel === "admin";
  // Close any socket we are replacing, and stop listening to it. Without this,
  // a superseded socket's close event arrives *after* the new one is already
  // working and flips the UI back to "reconnecting" -- leaving the map live and
  // correct but the compose button permanently disabled.
  const previous = socket;
  if (previous) {
    previous.onopen = previous.onmessage = previous.onclose = previous.onerror = null;
    try { previous.close(); } catch { /* already gone */ }
  }

  const ws = new WebSocket(wsUrl(admin ? "/ws/admin" : "/ws"));
  socket = ws;

  ws.onopen = () => {
    if (socket !== ws) return;
    lastFrameAt = Date.now();
    startWatchdog();
    send(admin
      ? { t: "hello", protocol: PROTOCOL, code: state.adminCode }
      : { t: "hello", protocol: PROTOCOL, id: state.identity });
  };

  ws.onmessage = (ev) => {
    if (socket !== ws) return;          // a stale socket must not drive the view
    lastFrameAt = Date.now();           // any frame, heartbeat included, is life
    let frame;
    try { frame = JSON.parse(ev.data); } catch { return; }
    handle(frame);
  };

  ws.onclose = () => {
    if (socket !== ws || closing) return;
    setConn("warn", "재연결 중…");
    retryTimer = setTimeout(connect, backoff);
    backoff = Math.min(8000, backoff * 2);      // 500ms -> 8s, then hold
  };

  ws.onerror = () => { /* a close event always follows; handled there */ };
}

function handle(frame) {
  switch (frame.t) {
    case "hello_ok": {
      backoff = 500;
      // A reconnect gets a fresh snapshot from the server, so anything the
      // client accumulated while it was away is replaced rather than merged.
      state.rev = frame.rev; state.layoutRev = frame.layout_rev;
      state.targets = frame.targets || [];
      state.targetIndex = new Map(state.targets.map((t, i) => [t.id, i]));
      state.roster = frame.roster || [];
      state.defaults = { source: frame.default_source || "ai",
                         week: frame.default_week || 1,
                         maxChars: frame.max_text_chars || 1000 };
      setConn("ok", frame.channel === "admin" ? "관리자 채널" : "연결됨");
      show("main");
      $("admin-badge").hidden = frame.channel !== "admin";
      $("adminpanel").hidden = frame.channel !== "admin";
      renderLegend();
      document.dispatchEvent(new CustomEvent("atlas:hello", { detail: frame }));
      autoEnableViewer();
      break;
    }
    case "snapshot": {
      // Below roughly a hundred and fifty opinions the server legitimately sends
      // snapshots rather than deltas, because a full recompute really does move
      // most points. Without diffing here, a new opinion arriving early in a
      // session -- which is every opinion in the first half hour -- would appear
      // with no animation at all, losing exactly the live moment the tool is for.
      const known = new Set(atlas.points.keys());
      const fresh = known.size ? frame.points.filter(p => !known.has(p.id)) : [];
      state.rev = frame.rev; state.layoutRev = frame.layout_rev;
      atlas.clear();
      atlas.setPoints(frame.points);
      if (fresh.length) atlas.markEntering(fresh.map(p => p.id));
      if (!atlas.fitted) atlas.fit(); else atlas.render();
      renderStatus();
      document.dispatchEvent(new CustomEvent("atlas:data"));
      break;
    }
    case "delta": {
      if (frame.from_rev !== state.rev) {
        // A gap. Ask for the whole thing rather than guessing at what was missed.
        send({ t: "resync", have_rev: state.rev });
        return;
      }
      state.rev = frame.rev; state.layoutRev = frame.layout_rev;
      atlas.upsert(frame.added || [], { entering: true });
      atlas.move(frame.moved || []);
      if (!atlas.fitted) atlas.fit(); else atlas.render();
      renderStatus();
      $("status-msg").textContent = frame.added?.length
        ? `새 의견 ${frame.added.length}건` : "";
      document.dispatchEvent(new CustomEvent("atlas:data"));
      break;
    }
    case "ack":
      state.rev = Math.max(state.rev, frame.rev);
      document.dispatchEvent(new CustomEvent("atlas:ack", { detail: frame }));
      break;
    case "viewer_refresh":
      if (viewerOn) viewer.refresh();
      break;
    case "ping":
      send({ t: "pong" });          // keeps the tunnel from calling us idle
      break;
    case "pong":
      break;
    case "neighbors":
      document.dispatchEvent(new CustomEvent("atlas:neighbors", { detail: frame }));
      break;
    case "error":
      document.dispatchEvent(new CustomEvent("atlas:error", { detail: frame }));
      if (frame.fatal) {
        closing = true;
        setConn("bad", frame.message);
        if (state.channel === "admin") { showAdminError(frame); show("admin"); }
        else { showGateError(frame); show("gate"); sessionStorage.removeItem("atlas.id"); }
      }
      break;
  }
}

/* ---------------- landing (PRD 5.1) ---------------- */
let gateAttempts = 0;

function showGateError(frame) {
  gateAttempts++;
  $("gate-error").hidden = false;
  $("gate-id").classList.add("input-bad");
  // Suspect a typo first; send them to find a person only after that has failed
  // a few times. Most failures really are typos, and walking to the front of the
  // room costs the presentation.
  $("gate-error-text").textContent = gateAttempts >= 3 && frame.code === "UNKNOWN_ID"
    ? `${frame.message} 강사·조교에게 명단 등록을 요청해 주세요.`
    : `${frame.message} ${frame.hint || ""}`.trim();
  $("gate-submit").disabled = false;
  $("gate-id").focus();
  $("gate-id").select();
}

function localGateError(message, hint = "") {
  $("gate-error").hidden = false;
  $("gate-error-text").textContent = `${message} ${hint}`.trim();
  $("gate-id").classList.add("input-bad");
  $("gate-id").focus();
}

$("gate-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const value = $("gate-id").value.trim();
  if (!value) {
    // The form is `novalidate` so this message is ours and in Korean; the
    // browser's own is English, on an otherwise Korean form. More importantly a
    // whitespace-only id passes `required` and used to return silently -- the
    // button was pressed and nothing at all happened, which reads as a broken
    // page rather than as a rejected input.
    localGateError("내 ID를 입력해 주세요.");
    return;
  }
  state.channel = "participant";
  state.identity = value;
  $("gate-error").hidden = true;
  $("gate-id").classList.remove("input-bad");
  $("gate-submit").disabled = true;
  setConn("warn", "연결 중…");
  try { sessionStorage.setItem("atlas.id", value); } catch { /* ignore */ }
  connect();
});

/* ---------------- admin gate (PRD 5.6) ---------------- */
function showAdminError(frame) {
  $("admin-error").hidden = false;
  $("admin-error-text").textContent = frame.message;
  $("admin-code").classList.add("input-bad");
  $("admin-code").focus();
}

$("admin-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (!$("admin-code").value.trim()) {
    $("admin-error").hidden = false;
    $("admin-error-text").textContent = "접근 코드를 입력해 주세요.";
    $("admin-code").classList.add("input-bad");
    $("admin-code").focus();
    return;
  }
  state.channel = "admin";
  state.adminCode = $("admin-code").value;
  $("admin-error").hidden = true;
  $("admin-code").classList.remove("input-bad");
  setConn("warn", "연결 중…");
  connect();
});
$("admin-back").addEventListener("click", (e) => {
  e.preventDefault();
  location.hash = "";
  location.reload();
});

/* ---------------- boot ---------------- */
mountCompose({ state, send, atlas });
mountAdmin({ state, send, atlas, renderStatus });
document.addEventListener("atlas:data", renderStatus);
// The server updates its DuckDB relation before it broadcasts, so by the time a
// frame lands here the rows behind the viewer are already the new ones.
document.addEventListener("atlas:data", () => { if (viewerOn) viewer.refresh(); });

if (location.hash === "#admin") {
  show("admin");
  $("admin-code").focus();
} else {
  let remembered = null;
  try { remembered = sessionStorage.getItem("atlas.id"); } catch { /* ignore */ }
  if (remembered) {
    // Same session, no re-entry (PRD 5.1).
    state.identity = remembered;
    setConn("warn", "연결 중…");
    connect();
    show("main");
  } else {
    show("gate");
    $("gate-id").focus();
  }
}
