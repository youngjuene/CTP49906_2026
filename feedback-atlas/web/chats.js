// The chat-trajectory map. One conversation at a time is drawn as a path through
// its sentences in the order they were said; every other conversation stays on
// the map, greyed, so the path is read against the whole corpus.
//
// Everything students and the model wrote reaches the DOM through textContent,
// never innerHTML: it is untrusted input that happens to be ours.

const NS = "http://www.w3.org/2000/svg";
// The classroom page's theme key (app-context.js storageKey at base "/"), so both
// pages follow the same light/dark choice.
import { arc, createView3d } from "./chats3d.js";

const THEME_KEY = "atlas.theme:classroom";
const COLOR_KEY = "atlas.chats.color-by";
const VIEW_KEY = "atlas.chats.view";

const $ = (id) => document.getElementById(id);

// "/" used to be the opinion map, whose "#admin" bookmark opened admin mode.
if (location.hash === "#admin") location.replace("/admin");
const state = {
  data: null, conv: null, units: [], route: [], cur: 0, showStatus: true,
  playing: null, glide: null, disp: null, routeDisp: null,
  colorBy: "turn", clusterColor: new Map(),
  // 3D view: on when the build has a 3D layout and the reader has not chosen 2D.
  view3d: false, disp3: null, routeDisp3: null, clusters: [],
  view: { cx: 0, cy: 0, scale: 1 }, fitScale: 1, size: [0, 0], drawn: null, els: new Map(),
};

try {
  const theme = localStorage.getItem(THEME_KEY);
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
} catch { /* storage blocked: follow the system */ }

// ---------- loading ----------

function showNotice(message) {
  $("app").hidden = true;
  $("notice").hidden = false;
  $("notice-text").textContent = message;
}

async function boot() {
  let res;
  try {
    res = await fetch("/api/chats", { cache: "no-store" });
  } catch {
    return showNotice("서버에 연결하지 못했습니다. 잠시 후 새로고침해 주세요.");
  }
  if (res.ok) return start(await res.json());
  let err = null;
  try { err = await res.json(); } catch { /* not JSON */ }
  showNotice(err ? `${err.message} ${err.hint || ""}`.trim()
                 : "대화 지도를 열 수 없습니다. 잠시 후 새로고침해 주세요.");
}

// ---------- colour ----------

function hexToRgb(hex) {
  const h = hex.trim().replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}
// Read the theme's ramp once per render, not once per point.
function rampScale() {
  const css = getComputedStyle(document.documentElement);
  const a = hexToRgb(css.getPropertyValue("--ramp-early"));
  const b = hexToRgb(css.getPropertyValue("--ramp-late"));
  return (t) => {
    const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
}

// ---------- map ----------

const svg = $("map");
// Three strengths of path. Consecutive sentences genuinely land far apart, so
// the full sentence path is a tangle by mid-conversation; it stays as faint
// context. The current turn is drawn strong, sentence by sentence. And the turn
// route -- one stop per turn, at the centre of that turn's sentences -- shows
// where the conversation as a whole travelled. It grows with playback: only the
// turns already reached are drawn, so it never reads as a path nobody played.
const gPathAll = document.createElementNS(NS, "path");
const gPathWalked = document.createElementNS(NS, "path");
const gPathTurn = document.createElementNS(NS, "path");
const gRoute = document.createElementNS(NS, "path");
const gPoints = document.createElementNS(NS, "g");
const gStops = document.createElementNS(NS, "g");
const cursor = document.createElementNS(NS, "circle");
const turnMark = document.createElementNS(NS, "text");
gPathAll.setAttribute("class", "path-all");
gPathWalked.setAttribute("class", "path-walked");
gPathTurn.setAttribute("class", "path-turn");
gRoute.setAttribute("class", "route");
cursor.setAttribute("class", "cursor");
cursor.setAttribute("r", "9");
turnMark.setAttribute("class", "turn-mark");
// Everything placed in pixels by renderStatic sits in gWorld (and gStops). A pan
// that does not change the zoom -- a drag, or the view following the cursor --
// only translates these two groups; the 800 points are re-placed once, after.
// Cluster keyword labels ride with the points, so a pan moves them together.
const gLabels = document.createElementNS(NS, "g");
const gWorld = document.createElementNS(NS, "g");
gWorld.append(gPathAll, gPathWalked, gPoints, gLabels);
svg.append(gWorld);
$("overlay").append(gRoute, gStops, gPathTurn, cursor, turnMark);

const $route = document.getElementById("show-route");

function measure() {
  const r = svg.getBoundingClientRect();
  state.size = [r.width, r.height];
}
const size = () => state.size;
const sx = (x) => state.size[0] / 2 + (x - state.view.cx) * state.view.scale;
const sy = (y) => state.size[1] / 2 - (y - state.view.cy) * state.view.scale;

function shiftDrawn() {
  const d = state.drawn;
  if (!d) return;
  const dx = (d.cx - state.view.cx) * d.scale;
  const dy = (state.view.cy - d.cy) * d.scale;
  const t = `translate(${dx.toFixed(1)} ${dy.toFixed(1)})`;
  gWorld.setAttribute("transform", t);
  gStops.setAttribute("transform", t);
}

// The 1st-99th percentile box, not the full extent: one stray sentence would
// otherwise set the frame and leave the conversation a smudge in a corner. The
// outliers stay reachable by panning.
function quantile(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]; }
function fit() {
  const pts = state.data.points;
  const xs = pts.map((p) => p.x).sort((a, b) => a - b);
  const ys = pts.map((p) => p.y).sort((a, b) => a - b);
  const [minX, maxX, minY, maxY] = [quantile(xs, 0.01), quantile(xs, 0.99), quantile(ys, 0.01), quantile(ys, 0.99)];
  measure();
  const [w, h] = size();
  state.view.cx = (minX + maxX) / 2;
  state.view.cy = (minY + maxY) / 2;
  state.view.scale = 0.9 * Math.min(w / Math.max(maxX - minX, 1e-6), h / Math.max(maxY - minY, 1e-6));
  state.fitScale = state.view.scale;
}

function shape(p) {
  const el = document.createElementNS(NS, p.kind === "status" ? "path"
    : p.speaker === "student" ? "rect" : "circle");
  const title = document.createElementNS(NS, "title");
  title.textContent = p.text;
  el.append(title);
  el.dataset.id = p.id;
  return el;
}

function place(el, p) {
  const x = sx(p.x), y = sy(p.y);
  if (el.tagName === "circle") { el.setAttribute("cx", x); el.setAttribute("cy", y); el.setAttribute("r", 3.6); }
  else if (el.tagName === "rect") {
    el.setAttribute("x", x - 3.8); el.setAttribute("y", y - 3.8);
    el.setAttribute("width", 7.6); el.setAttribute("height", 7.6);
  } else el.setAttribute("d", `M${x} ${y - 4.5}L${x + 4.5} ${y}L${x} ${y + 4.5}L${x - 4.5} ${y}Z`);
}

function buildPoints() {
  gPoints.replaceChildren();
  state.els.clear();
  state.drawn = null;            // new elements: nothing is placed yet
  // Other conversations first, so the selected one is drawn on top.
  const order = [...state.data.points].sort((a, b) =>
    (a.conversation === state.conv.id) - (b.conversation === state.conv.id));
  for (const p of order) {
    const el = shape(p);
    state.els.set(p.id, el);
    gPoints.append(el);
  }
}

// Two layers of drawing. renderStatic places every point and the settled parts
// of the paths; it runs on a step, a zoom or a resize. renderMotion draws only
// what moves -- the cursor, the growing end of the current turn's line, and the
// growing end of the turn route -- and runs every animation frame, so a glide
// does not re-place 800 points sixty times a second.
function render() {
  renderStatic();
  renderMotion();
}

// Write an attribute or style only when it differs from what was last written:
// a step changes a handful of points, and touching all 800 costs a frame.
function setOnce(el, key, value, write) {
  if (el[key] === value) return;
  el[key] = value;
  write(value);
}

function renderStatic() {
  if (state.view3d) return draw3d();
  measure();
  const turns = Math.max(state.conv.turns - 1, 1);
  const here = state.units[state.cur];
  const v = state.view;
  const placed = state.drawn && state.drawn.cx === v.cx && state.drawn.cy === v.cy
    && state.drawn.scale === v.scale && state.drawn.w === state.size[0] && state.drawn.h === state.size[1];
  const color = rampScale();
  for (const p of state.data.points) {
    const el = state.els.get(p.id);
    const mine = p.conversation === state.conv.id;
    const hidden = p.kind === "status" && !state.showStatus;
    if (!placed) place(el, p);
    setOnce(el, "_display", hidden ? "none" : "", (d) => { el.style.display = d; });
    if (hidden) continue;
    let cls = p.kind === "status" ? "pt status" : "pt";
    if (!mine) cls += " other";
    else if (here && p.seq > here.seq) cls += " ahead";
    setOnce(el, "_cls", cls, (c) => el.setAttribute("class", c));
    const paint = !mine ? ""
      : state.colorBy === "cluster" && p.cluster != null ? state.clusterColor.get(p.cluster)
      : color((p.turn - 1) / turns);
    setOnce(el, "_paint", `${p.kind}:${paint}`, () => {
      el.style.fill = p.kind === "status" ? "" : paint;
      el.style.stroke = p.kind === "status" ? paint : "";
    });
  }
  if (!placed) {
    for (const label of gLabels.children) {
      label.setAttribute("x", sx(label._at.x)); label.setAttribute("y", sy(label._at.y));
    }
    state.drawn = { cx: v.cx, cy: v.cy, scale: v.scale, w: state.size[0], h: state.size[1] };
    gWorld.removeAttribute("transform");
  }
  gLabels.style.display = state.colorBy === "cluster" ? "" : "none";
  gStops.removeAttribute("transform");
  gPathAll.setAttribute("d", line(state.units));
  const start = here ? turnStart(here.turn) : 0;
  // The walked path stops where the current turn begins; the current turn's
  // strong line is drawn in renderMotion.
  gPathWalked.setAttribute("d", line(state.units.slice(0, start + 1)));

  const at = here ? here.turn : 0;
  const showRoute = $route.checked;
  gStops.replaceChildren();
  if (showRoute) {
    for (const c of state.route) {
      if (c.turn > at) break;          // route is sorted by turn
      const [X, Y] = [sx(c.x), sy(c.y)];
      const dot = document.createElementNS(NS, "circle");
      dot.setAttribute("class", `stop${c.turn === at ? " now" : ""}`);
      dot.setAttribute("cx", X); dot.setAttribute("cy", Y); dot.setAttribute("r", 7);
      const label = document.createElementNS(NS, "text");
      label.setAttribute("class", "stop-label");
      label.setAttribute("x", X); label.setAttribute("y", Y + 3.5);
      label.textContent = c.turn;
      gStops.append(dot, label);
    }
  }
  if (here) turnMark.textContent = `T${here.turn}`;
}

const line = (list) => list.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`).join("");

function renderMotion() {
  if (state.view3d) return draw3d();
  const here = state.units[state.cur];
  if (!here) return;
  const at = state.disp || here;
  const start = turnStart(here.turn);
  // The current turn's line runs from the previous turn's last unit up to the
  // unit before this one, then to wherever the cursor is right now -- so the
  // newest segment grows as the cursor glides along it.
  gPathTurn.setAttribute("d", line([
    ...state.units.slice(Math.max(start - 1, 0), state.cur), at]));
  cursor.setAttribute("cx", sx(at.x)); cursor.setAttribute("cy", sy(at.y));
  turnMark.setAttribute("x", sx(at.x) + 12); turnMark.setAttribute("y", sy(at.y) - 10);

  if ($route.checked) {
    const stop = state.route.find((c) => c.turn === here.turn);
    const head = state.routeDisp || stop;
    gRoute.setAttribute("d", line([...state.route.filter((c) => c.turn < here.turn), head]));
  } else {
    gRoute.setAttribute("d", "");
  }
}

// ---------- motion ----------

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const lerp = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
const speed = () => Number($("speed").value) || 1;

function cancelGlide() {
  if (!state.glide) return;
  cancelAnimationFrame(state.glide.raf);
  const done = state.glide.done;
  state.glide = null;
  done();
}

// Glide the cursor (and, on a new turn, the head of the turn route) from where
// it is drawn now to the current unit. Resolves when it arrives, or at once
// when it is cancelled by the next step.
function glide(ms) {
  cancelGlide();
  const here = state.units[state.cur];
  const stopNow = state.route.find((c) => c.turn === here.turn);
  const from = state.disp || here;
  const routeFrom = state.routeDisp || stopNow;
  // In 3D the cursor travels the same SLERP arc the path is drawn with.
  const from3 = state.disp3 || here;
  const routeFrom3 = state.routeDisp3 || stopNow;
  if (reduceMotion.matches || ms <= 0) {
    state.disp = { x: here.x, y: here.y };
    state.routeDisp = stopNow;
    state.disp3 = here;
    state.routeDisp3 = stopNow;
    renderMotion();
    return Promise.resolve();
  }
  return new Promise((done) => {
    const t0 = performance.now();
    const frame = (now) => {
      const t = Math.min(1, (now - t0) / ms);
      const k = ease(t);
      state.disp = lerp(from, here, k);
      state.routeDisp = stopNow ? lerp(routeFrom, stopNow, k) : null;
      if (state.view3d) {
        state.disp3 = arc(from3, here, k);
        state.routeDisp3 = stopNow ? arc(routeFrom3, stopNow, k) : null;
        if ($("follow").checked) view3d.lookToward(state.disp3);
      } else if ($("follow").checked) follow();
      renderMotion();
      if (t < 1) state.glide.raf = requestAnimationFrame(frame);
      else { state.glide = null; done(); }
    };
    state.glide = { raf: requestAnimationFrame(frame), done };
  });
}

// When zoomed in, keep the cursor on screen by easing the view toward it once it
// nears an edge. Never at the fitted view: there everything is already visible,
// and a map that drifts on its own is harder to read than one that holds still.
// The pan only translates the drawn layer; points are re-placed at the next step.
function follow() {
  if (state.view.scale < state.fitScale * 1.2) return;
  const [w, h] = size();
  const x = sx(state.disp.x), y = sy(state.disp.y);
  const mx = w * 0.15, my = h * 0.15;
  let dx = 0, dy = 0;
  if (x < mx) dx = x - mx; else if (x > w - mx) dx = x - (w - mx);
  if (y < my) dy = y - my; else if (y > h - my) dy = y - (h - my);
  if (!dx && !dy) return;
  state.view.cx += (dx * 0.12) / state.view.scale;
  state.view.cy -= (dy * 0.12) / state.view.scale;
  shiftDrawn();
}

// ---------- panel ----------

const fmtTime = (ts) => ts ? new Date(ts).toLocaleString("ko-KR", {
  timeZone: "Asia/Seoul", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit",
}) : "";

function renderPanel() {
  const here = state.units[state.cur];
  if (!here) return;
  const who = here.speaker === "student" ? "학생" : "AI";
  $("meta").textContent =
    `턴 ${here.turn}/${state.conv.turns} · ${who}${here.kind === "status" ? " (상태)" : ""} · ${fmtTime(here.timestamp)}`;
  $("pos").textContent = `${state.cur + 1} / ${state.units.length}`;
  $("scrub").max = String(state.units.length - 1);
  $("scrub").value = String(state.cur);

  const list = $("turn");
  list.replaceChildren();
  let lastSpeaker = null, current = null;
  state.units.forEach((u, i) => {
    if (u.turn !== here.turn) return;
    if (u.speaker !== lastSpeaker) {
      const head = document.createElement("li");
      head.className = "who";
      head.textContent = u.speaker === "student" ? "학생" : "AI";
      list.append(head);
      lastSpeaker = u.speaker;
    }
    const li = document.createElement("li");
    li.textContent = u.text;
    if (u.kind === "status") li.classList.add("status");
    if (i === state.cur) { li.classList.add("here"); current = li; }
    li.addEventListener("click", () => go(i));
    list.append(li);
  });
  current?.scrollIntoView({ block: "nearest" });
}

// ---------- navigation ----------

const GLIDE_MS = 520;

// A step settles the map and panel at once and glides the cursor there; a
// jump (the slider, a new conversation) lands without a glide.
function go(i, { animate = true } = {}) {
  state.cur = Math.max(0, Math.min(state.units.length - 1, i));
  if (!animate) cancelGlide();
  renderStatic();
  renderPanel();
  return glide(animate ? GLIDE_MS / speed() : 0);
}
function turnStart(turn) { return state.units.findIndex((u) => u.turn === turn); }
function prevTurn() {
  const here = state.units[state.cur];
  const start = turnStart(here.turn);
  go(state.cur > start ? start : Math.max(turnStart(here.turn - 1), 0));
}
function nextTurn() {
  const next = turnStart(state.units[state.cur].turn + 1);
  if (next >= 0) go(next);
}
function stop() {
  state.playing = null;
  $("play").textContent = "▶ 재생";
}

// Time to read a unit before moving on: longer sentences hold longer. At 1x a
// typical 55-character sentence takes about 1.1 s including the glide.
const dwell = (u) => Math.min(1600, Math.max(250, 150 + 8 * u.text.length)) / speed();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function togglePlay() {
  if (state.playing) return stop();
  const run = {};
  state.playing = run;
  $("play").textContent = "⏸ 멈춤";
  if (state.cur >= state.units.length - 1) await go(0, { animate: false });
  while (state.playing === run) {
    await sleep(dwell(state.units[state.cur]));
    if (state.playing !== run) return;
    if (state.cur >= state.units.length - 1) return stop();
    await go(state.cur + 1);
  }
}

// One stop per turn: the mean position of that turn's sentences, prompt and
// reply together. Status lines are pinned to sentences and would only pull the
// centre toward their neighbour twice.
function buildRoute() {
  const byTurn = new Map();
  for (const p of state.data.points) {
    if (p.conversation !== state.conv.id || p.kind === "status") continue;
    const acc = byTurn.get(p.turn) || { x: 0, y: 0, x3: 0, y3: 0, z3: 0, n: 0 };
    acc.x += p.x; acc.y += p.y; acc.n += 1;
    acc.x3 += p.x3 ?? 0; acc.y3 += p.y3 ?? 0; acc.z3 += p.z3 ?? 0;
    byTurn.set(p.turn, acc);
  }
  state.route = [...byTurn.entries()].sort((a, b) => a[0] - b[0])
    .map(([turn, a]) => ({ turn, x: a.x / a.n, y: a.y / a.n,
                           x3: a.x3 / a.n, y3: a.y3 / a.n, z3: a.z3 / a.n }));
}

// Redraw everything at once, without the per-point fade, then let fades resume.
function instantly(redraw) {
  svg.classList.add("instant");
  redraw();
  requestAnimationFrame(() => requestAnimationFrame(() => svg.classList.remove("instant")));
}

function selectConversation(id) {
  instantly(() => selectConversationNow(id));
}

function selectConversationNow(id) {
  stop();
  state.conv = state.data.conversations.find((c) => c.id === id) || state.data.conversations[0];
  state.units = state.data.points
    .filter((p) => p.conversation === state.conv.id && (state.showStatus || p.kind !== "status"))
    .sort((a, b) => a.seq - b.seq);
  buildRoute();
  buildPoints();
  state.disp = null;
  state.routeDisp = null;
  state.disp3 = null;
  state.routeDisp3 = null;
  go(0, { animate: false });
}

$("prev").addEventListener("click", () => { stop(); go(state.cur - 1); });
$("next").addEventListener("click", () => { stop(); go(state.cur + 1); });
$("prev-turn").addEventListener("click", () => { stop(); prevTurn(); });
$("next-turn").addEventListener("click", () => { stop(); nextTurn(); });
$("play").addEventListener("click", togglePlay);
$("scrub").addEventListener("input", (e) => { stop(); go(Number(e.target.value), { animate: false }); });
$("conv").addEventListener("change", (e) => selectConversation(e.target.value));
$route.addEventListener("change", () => render());
$("show-status").addEventListener("change", (e) => {
  const seq = state.units[state.cur]?.seq ?? 0;
  state.showStatus = e.target.checked;
  state.units = state.data.points
    .filter((p) => p.conversation === state.conv.id && (state.showStatus || p.kind !== "status"))
    .sort((a, b) => a.seq - b.seq);
  // Stay where the reader was, or on the nearest unit before it.
  let i = state.units.findIndex((u) => u.seq >= seq);
  if (i < 0) i = state.units.length - 1;
  go(i, { animate: false });
});
document.addEventListener("keydown", (e) => {
  if ($("app").hidden || e.target.matches("input[type=text],input[type=password],select")) return;
  if (e.key === "ArrowRight") { stop(); e.shiftKey ? nextTurn() : go(state.cur + 1); e.preventDefault(); }
  else if (e.key === "ArrowLeft") { stop(); e.shiftKey ? prevTurn() : go(state.cur - 1); e.preventDefault(); }
  else if (e.key === " " && !e.target.matches("button")) { togglePlay(); e.preventDefault(); }
});

// ---------- pan, zoom, tap ----------

const pointers = new Map();
let dragFrom = null, pinchFrom = null, moved = 0;

svg.addEventListener("pointerdown", (e) => {
  svg.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  moved = 0;
  if (pointers.size === 1) { dragFrom = { x: e.clientX, y: e.clientY, ...state.view }; svg.classList.add("dragging"); }
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    pinchFrom = { d: Math.hypot(a.x - b.x, a.y - b.y), scale: state.view.scale };
  }
});
svg.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 2 && pinchFrom) {
    const [a, b] = [...pointers.values()];
    state.view.scale = pinchFrom.scale * Math.hypot(a.x - b.x, a.y - b.y) / pinchFrom.d;
    moved = Infinity;
    render();
  } else if (dragFrom) {
    const dx = e.clientX - dragFrom.x, dy = e.clientY - dragFrom.y;
    moved = Math.max(moved, Math.hypot(dx, dy));
    state.view.cx = dragFrom.cx - dx / state.view.scale;
    state.view.cy = dragFrom.cy + dy / state.view.scale;
    shiftDrawn();
    renderMotion();
  }
});
function release(e) {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinchFrom = null;
  if (pointers.size === 0) {
    if (dragFrom && moved >= 4) render();     // settle the drag: re-place once
    dragFrom = null; svg.classList.remove("dragging");
    if (moved < 4 && e.type === "pointerup") tap(e);
  }
}
svg.addEventListener("pointerup", release);
svg.addEventListener("pointercancel", release);
svg.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = svg.getBoundingClientRect();
  const k = Math.exp(-e.deltaY * 0.0015);
  // Zoom about the cursor: the map point under it stays under it.
  const mx = (e.clientX - r.left - r.width / 2) / state.view.scale + state.view.cx;
  const my = -(e.clientY - r.top - r.height / 2) / state.view.scale + state.view.cy;
  state.view.scale *= k;
  state.view.cx = mx - (mx - state.view.cx) / k;
  state.view.cy = my - (my - state.view.cy) / k;
  render();
}, { passive: false });
svg.addEventListener("dblclick", () => { fit(); render(); });

function tap(e) {
  const r = svg.getBoundingClientRect();
  const px = e.clientX - r.left, py = e.clientY - r.top;
  // Nearest point of the selected conversation within reach; a tap on another
  // conversation's point switches to it.
  let best = null, bestD = 14;
  for (const p of state.data.points) {
    if (p.kind === "status" && !state.showStatus) continue;
    const d = Math.hypot(sx(p.x) - px, sy(p.y) - py) - (p.conversation === state.conv.id ? 2 : 0);
    if (d < bestD) { best = p; bestD = d; }
  }
  if (!best) return;
  stop();
  if (best.conversation !== state.conv.id) {
    $("conv").value = best.conversation;
    selectConversation(best.conversation);
  }
  go(state.units.findIndex((u) => u.id === best.id));
}

new ResizeObserver(() => {
  if (!state.data || !state.conv) return;
  if (state.view3d) view3d.resize();
  render();
}).observe(svg.parentElement);

// ---------- 3D ----------
// The same scene as the flat map, handed to chats3d.js to draw: which points are
// shown and how each is coloured, a short fading trail behind the gliding
// cursor, and the turn route grown to the current turn. Fewer lines than the
// flat map on purpose: arcs loop around the centre and pile up fast.

const view3d = createView3d($("map3d"), {
  onTap(p) {
    stop();
    if (p.conversation !== state.conv.id) {
      $("conv").value = p.conversation;
      selectConversation(p.conversation);
    }
    go(state.units.findIndex((u) => u.id === p.id));
  },
});

const TRAIL_3D = 6;               // steps of trail behind the cursor in 3D

function draw3d() {
  const here = state.units[state.cur];
  if (!here) return;
  const turns = Math.max(state.conv.turns - 1, 1);
  const ramp = rampScale();
  const css = getComputedStyle(document.documentElement);
  const grey = css.getPropertyValue("--other").trim() || "#94a3b8";
  const mine = (p) => p.conversation === state.conv.id;
  view3d.draw({
    points: state.data.points,
    mine,
    style(p) {
      if (p.kind === "status" && !state.showStatus) return null;
      if (!mine(p)) return { colour: grey, alpha: p.kind === "status" ? 0.2 : 0.35 };
      const colour = state.colorBy === "cluster" && p.cluster != null
        ? state.clusterColor.get(p.cluster) : ramp((p.turn - 1) / turns);
      return { colour, alpha: p.seq > here.seq ? 0.45 : 1 };
    },
    trail: state.units.slice(Math.max(state.cur - TRAIL_3D, 0), state.cur),
    cursor: state.disp3 || here,
    turnMark: `T${here.turn}`,
    route: $route.checked ? state.route.filter((c) => c.turn < here.turn) : [],
    routeHead: $route.checked ? (state.routeDisp3 || state.route.find((c) => c.turn === here.turn)) : null,
    stops: $route.checked
      ? state.route.filter((c) => c.turn <= here.turn).map((c) => ({ ...c, now: c.turn === here.turn }))
      : [],
    labels: state.colorBy === "cluster" ? state.clusters.filter((c) => c.z3 != null) : [],
  });
}

function setView(mode, { redraw = true } = {}) {
  state.view3d = mode === "3d";
  $("view-mode").value = mode;
  // SVG elements have no `hidden` property, only the attribute.
  svg.toggleAttribute("hidden", state.view3d);
  $("overlay").toggleAttribute("hidden", state.view3d);
  $("map3d").hidden = !state.view3d;
  $("map-note").textContent = `문장 ${state.data.points.length}개 · 위치는 의미 유사도, 선은 대화 순서 · ` +
    (state.view3d ? "끌어서 회전 · 휠로 확대 · 더블클릭으로 처음 보기" : "더블클릭으로 전체 보기");
  if (state.view3d) view3d.resize();
  if (!redraw || !state.conv) return;
  state.drawn = null;                 // the flat map re-places everything when it returns
  render();
}

$("view-mode").addEventListener("change", (e) => {
  try { localStorage.setItem(VIEW_KEY, e.target.value); } catch { /* storage blocked */ }
  setView(e.target.value);
});

// ---------- start ----------

// ---------- clusters ----------
// Topic clusters come from the build (scripts/chat_clusters.py): one colour from
// matplotlib's default cycle each, with neighbouring clusters kept visually apart.

function buildClusters(clusters) {
  state.clusters = clusters;
  state.clusterColor = new Map(clusters.map((c) => [c.id, c.color]));
  const legend = $("cluster-legend");
  legend.replaceChildren();
  gLabels.replaceChildren();
  for (const c of clusters) {
    const row = document.createElement("div");
    row.className = "row";
    const sw = document.createElement("span");
    sw.className = "sw";
    sw.style.background = c.color;
    const name = document.createElement("span");
    name.textContent = c.label;
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = c.size;
    row.append(sw, name, n);
    legend.append(row);

    const label = document.createElementNS(NS, "text");
    label.setAttribute("class", "cluster-label");
    label.style.fill = c.color;
    label.textContent = c.label;
    label._at = c;
    gLabels.append(label);
  }
  $("color-by-wrap").hidden = !clusters.length;
  let saved = null;
  try { saved = localStorage.getItem(COLOR_KEY); } catch { /* storage blocked */ }
  setColorBy(clusters.length && saved !== "turn" ? "cluster" : "turn");
}

function setColorBy(mode) {
  state.colorBy = mode;
  $("color-by").value = mode;
  $("legend-turn").hidden = mode !== "turn";
  $("cluster-legend").hidden = mode !== "cluster";
  if (state.conv) instantly(renderStatic);
}

$("color-by").addEventListener("change", (e) => {
  try { localStorage.setItem(COLOR_KEY, e.target.value); } catch { /* storage blocked */ }
  setColorBy(e.target.value);
});

function start(data) {
  if (!data.points?.length) return showNotice("아직 올라온 대화가 없습니다.");
  state.data = data;
  $("notice").hidden = true;
  $("app").hidden = false;
  const names = Object.fromEntries((data.targets || []).map((t) => [t.id, t.display_name]));
  const select = $("conv");
  select.replaceChildren();
  for (const c of data.conversations) {
    const opt = document.createElement("option");
    opt.value = c.id;
    const nn = c.id.split("__").pop();
    opt.textContent = `${names[c.target_id] || c.target_id} · ${c.week}주차 · #${nn} (${c.turns}턴)`;
    select.append(opt);
  }
  buildClusters(data.clusters || []);
  fit();
  const has3d = data.points.every((p) => p.z3 != null);
  $("view-mode-wrap").hidden = !has3d;
  let saved = null;
  try { saved = localStorage.getItem(VIEW_KEY); } catch { /* storage blocked */ }
  setView(has3d && saved !== "2d" ? "3d" : "2d", { redraw: false });
  selectConversation(data.conversations[0].id);
}

boot();
