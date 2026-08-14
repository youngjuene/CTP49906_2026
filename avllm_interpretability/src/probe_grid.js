// probe_grid.js -- the whole 36 x 248 logit-lens probe surface at once.
//
// Plain ES2020, zero imports: molab's CSP blocks third-party hosts, so no React,
// no d3, no CDN. Everything below is hand-rolled canvas + DOM on purpose.
//
// The one invariant to preserve while editing: this file NEVER calls
// model.set()/model.save_changes(). A js->py trait write would dirty the widget
// and re-run every marimo cell that references it, and the neighbouring cells
// hold a model on the GPU. Active layer, pinned column and the junk toggle are
// client-side state and stay that way.

const CONTENT = 0;
const JUNK = 1;
const UNDECODABLE = 2;

const GUTTER_L = 52; // room for layer labels
const PAD_T = 6;
const PAD_R = 8;
const AXIS_H = 22; // time axis under the grid
const ROW_H = 11; // one layer row
const MIN_W = 300;

// Fallbacks used when the CSS custom properties are missing (a host that strips
// _css still gets a readable widget).
const FALLBACK = {
  content: "#0f766e",
  junk: "#c2c7ce",
  undecodable: "#dc2626",
  settled: "rgba(17,24,39,0.72)",
  grid: "rgba(120,120,120,0.35)",
  fg: "#1f2328",
  muted: "#6b7280",
  pin: "#b45309",
};

// anywidget versions differ in what a traitlets.Bytes lands as: a DataView, a
// raw ArrayBuffer, or a typed array. Normalise all three.
function toBytes(value) {
  if (!value) return null;
  if (value instanceof DataView || ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  return null;
}

function escapeToken(text) {
  return String(text)
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t");
}

// Chips must show whitespace-only tokens as *something*; a blank chip reads as a
// rendering bug rather than as the finding.
function chipLabel(text) {
  const raw = String(text);
  if (raw === "") return "∅";
  if (!raw.trim()) {
    return raw
      .replace(/ /g, "␣")
      .replace(/\n/g, "⏎")
      .replace(/\t/g, "⇥");
  }
  return escapeToken(raw);
}

function readPalette(el) {
  const cs = getComputedStyle(el);
  const pick = (name, fallback) => {
    const value = cs.getPropertyValue(name);
    return value && value.trim() ? value.trim() : fallback;
  };
  // currentColor is not a valid canvas fillStyle, so resolve it here.
  const ink = cs.color || FALLBACK.fg;
  return {
    content: pick("--ctp-pg-content", FALLBACK.content),
    junk: pick("--ctp-pg-junk", FALLBACK.junk),
    undecodable: pick("--ctp-pg-undecodable", FALLBACK.undecodable),
    settled: pick("--ctp-pg-settled", FALLBACK.settled),
    grid: pick("--ctp-pg-grid", FALLBACK.grid),
    muted: pick("--ctp-pg-muted", FALLBACK.muted),
    pin: pick("--ctp-pg-pin", FALLBACK.pin),
    fg: pick("--ctp-pg-fg", ink),
  };
}

function div(parent, className, text) {
  const node = document.createElement("div");
  node.className = className;
  if (text != null) node.textContent = text;
  parent.appendChild(node);
  return node;
}

function showError(el, err) {
  el.textContent = "";
  const box = document.createElement("pre");
  box.className = "pg-error";
  // Inside molab's locked-down iframe the console is not reachable, so the
  // error has to become visible content or the widget just looks blank.
  box.textContent =
    "probe_grid.js failed to render:\n" +
    (err && err.stack ? err.stack : String(err));
  el.appendChild(box);
}

function build(model, el) {
  const layerNames = model.get("layer_names") || [];
  const positions = model.get("positions") || [];
  const vocab = model.get("vocab") || [];
  const vocabClass = toBytes(model.get("vocab_class"));
  const indexBytes = toBytes(model.get("token_index"));
  const spp = Number(model.get("seconds_per_position")) || 0.04;
  const junkDefinition = model.get("junk_definition") || "";
  const caveat = model.get("caveat") || "";

  const nLayers = layerNames.length;
  const nPos = positions.length;
  const needed = 2 * nPos * nLayers;
  if (!nLayers || !nPos || !indexBytes || indexBytes.byteLength < needed) {
    div(el, "pg-empty", "No logit-lens probe grid yet — run the sweep first.");
    return () => {};
  }

  // Explicit little-endian reads: the Python side packs uint16 LE regardless of
  // the host's native order, so never trust the platform default here.
  const view = new DataView(
    indexBytes.buffer,
    indexBytes.byteOffset,
    indexBytes.byteLength
  );
  const grid = new Uint16Array(nPos * nLayers);
  for (let i = 0; i < grid.length; i++) grid[i] = view.getUint16(i * 2, true);

  const at = (p, k) => grid[p * nLayers + k];
  const classOf = (slot) => (vocabClass && slot < vocabClass.length ? vocabClass[slot] : CONTENT);
  const finalOf = new Uint16Array(nPos);
  for (let p = 0; p < nPos; p++) finalOf[p] = at(p, nLayers - 1);

  // Per-layer stats, computed once from the same arrays the canvas draws from,
  // so the readout can never drift from the picture.
  const stats = [];
  for (let k = 0; k < nLayers; k++) {
    const counts = new Map();
    let junk = 0;
    let undecodable = 0;
    let settled = 0;
    for (let p = 0; p < nPos; p++) {
      const slot = at(p, k);
      counts.set(slot, (counts.get(slot) || 0) + 1);
      const cls = classOf(slot);
      if (cls === JUNK) junk++;
      else if (cls === UNDECODABLE) undecodable++;
      if (slot === finalOf[p]) settled++;
    }
    let modal = -1;
    let modalCount = 0;
    counts.forEach((count, slot) => {
      if (count > modalCount || (count === modalCount && slot < modal)) {
        modal = slot;
        modalCount = count;
      }
    });
    stats.push({
      unique: counts.size,
      junk,
      undecodable,
      settled,
      modal: modal >= 0 ? vocab[modal] : "",
      modalCount,
    });
  }

  // Discontinuities in the ABSOLUTE positions. They exist because
  // use_audio_in_video interleaves the streams, and marking them keeps the x
  // axis honest: it is ordinal, and these are where the ordinal and the
  // absolute index part company.
  const breaks = [];
  for (let p = 1; p < nPos; p++) {
    if (positions[p] !== positions[p - 1] + 1) breaks.push(p);
  }

  const hz = Math.round(1 / spp);
  const finalJunk = stats[nLayers - 1].junk;

  // ---------------------------------------------------------------- DOM ----
  const head = div(el, "pg-head");
  div(head, "pg-title", "Logit-lens probe grid");
  div(
    head,
    "pg-verdict",
    `final layer: ${finalJunk}/${nPos} junk · ${stats[nLayers - 1].unique} unique tokens`
  );
  const readout = div(el, "pg-readout", "");

  const canvasWrap = div(el, "pg-canvas-wrap");
  const canvas = document.createElement("canvas");
  canvas.className = "pg-canvas";
  canvas.setAttribute("role", "img");
  canvasWrap.appendChild(canvas);
  const tip = div(canvasWrap, "pg-tip");
  tip.hidden = true;

  const legend = div(el, "pg-legend");
  const swatch = (cls, label) => {
    const item = div(legend, "pg-legend-item");
    div(item, "pg-sw pg-sw-" + cls);
    div(item, "pg-legend-label", label);
  };
  swatch("content", "content");
  swatch("junk", "junk (all punctuation / space / symbol)");
  swatch("undecodable", "undecodable (U+FFFD)");
  swatch("settled", "ring = already equals this position's final token");

  const controls = div(el, "pg-controls");
  const toggleLabel = document.createElement("label");
  toggleLabel.className = "pg-toggle";
  const toggle = document.createElement("input");
  toggle.type = "checkbox";
  toggleLabel.appendChild(toggle);
  toggleLabel.appendChild(document.createTextNode(" show junk undimmed"));
  controls.appendChild(toggleLabel);
  div(
    controls,
    "pg-hint",
    "drag the grid to scrub layers · click a column to pin its trajectory"
  );

  const chipsHead = div(el, "pg-chips-head", "");
  const chips = div(el, "pg-chips");
  const chipEls = [];
  for (let p = 0; p < nPos; p++) {
    const chip = document.createElement("span");
    chip.className = "pg-chip";
    chip.dataset.pos = String(p);
    chips.appendChild(chip);
    chipEls.push(chip);
  }

  const pinPanel = div(el, "pg-pin-panel");
  pinPanel.hidden = true;
  const pinHead = div(pinPanel, "pg-pin-head", "");
  const pinStrip = div(pinPanel, "pg-pin-strip");
  const pinRows = [];
  for (let k = 0; k < nLayers; k++) {
    const row = div(pinStrip, "pg-pin-row");
    div(row, "pg-pin-layer", layerNames[k]);
    const bar = div(row, "pg-pin-bar");
    const token = div(row, "pg-pin-token", "");
    pinRows.push({ row, bar, token });
  }

  const notes = div(el, "pg-notes");
  div(
    notes,
    "pg-note",
    `x axis is the ORDINAL audio position (0–${nPos - 1}) at ${hz} Hz, not the ` +
      `absolute token index: the ${nPos} positions arrive in ${breaks.length + 1} ` +
      `run(s) (${positions[0]}–${positions[nPos - 1]}), and the dashed rules mark ` +
      `where the absolute index jumps. Times are frame ends — (ordinal + 1) × ${spp} s, ` +
      `so the last frame lands at ${(nPos * spp).toFixed(2)} s.`
  );
  div(notes, "pg-note pg-note-def", junkDefinition);
  div(notes, "pg-note pg-note-caveat", caveat);

  // ------------------------------------------------------------- state ----
  // Open on the junkiest layer rather than layer 0 or the final one: the point
  // of the widget is the degeneracy, so it should be on screen before anyone
  // touches a control. (On the committed CSV that is Layer_34, 245/248 junk.)
  let activeLayer = 0;
  for (let k = 1; k < nLayers; k++) {
    if (stats[k].junk > stats[activeLayer].junk) activeLayer = k;
  }
  let pinned = -1;
  let cssW = MIN_W;
  let cssH = PAD_T + nLayers * ROW_H + AXIS_H;
  let cellW = 1;
  let frame = 0;

  const reduceMotion =
    typeof matchMedia === "function" &&
    matchMedia("(prefers-reduced-motion: reduce)").matches;

  const timeLabel = (p) => `${((p + 1) * spp).toFixed(2)} s`;

  function describe(p, k) {
    const slot = at(p, k);
    const cls = classOf(slot);
    const kind = cls === JUNK ? "junk" : cls === UNDECODABLE ? "undecodable" : "content";
    return (
      `pos ${positions[p]} · t≈${timeLabel(p)} ` +
      `(audio token ${p + 1}/${nPos} @ ${hz} Hz) · ` +
      `'${escapeToken(vocab[slot])}' · ${kind}`
    );
  }

  // ------------------------------------------------------------ drawing ----
  function layout() {
    const width = canvasWrap.clientWidth || el.clientWidth || MIN_W;
    cssW = Math.max(MIN_W, Math.floor(width));
    cssH = PAD_T + nLayers * ROW_H + AXIS_H;
    cellW = (cssW - GUTTER_L - PAD_R) / nPos;
    // devicePixelRatio backing store, CSS-pixel drawing space: without this the
    // 1px rings and the axis text are mush on a HiDPI screen.
    const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function draw() {
    const ctx = layout();
    const pal = readPalette(el);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.textBaseline = "middle";

    const fillFor = [pal.content, pal.junk, pal.undecodable];
    const drawRing = cellW >= 2.2 && ROW_H >= 4;

    for (let k = 0; k < nLayers; k++) {
      const y = PAD_T + k * ROW_H;
      for (let p = 0; p < nPos; p++) {
        const slot = at(p, k);
        const x = GUTTER_L + p * cellW;
        ctx.fillStyle = fillFor[classOf(slot)] || pal.content;
        // +0.6 so neighbouring cells butt together instead of showing seams at
        // fractional widths.
        ctx.fillRect(x, y, cellW + 0.6, ROW_H - 1);
        if (slot === finalOf[p]) {
          ctx.strokeStyle = pal.settled;
          ctx.lineWidth = 1;
          if (drawRing) {
            ctx.strokeRect(x + 0.5, y + 0.5, Math.max(1, cellW - 1), ROW_H - 2);
          } else {
            // Too narrow for a ring: a top+bottom tick still reads as texture.
            ctx.beginPath();
            ctx.moveTo(x, y + 0.5);
            ctx.lineTo(x + cellW, y + 0.5);
            ctx.moveTo(x, y + ROW_H - 1.5);
            ctx.lineTo(x + cellW, y + ROW_H - 1.5);
            ctx.stroke();
          }
        }
      }
    }

    // Run boundaries in the absolute positions.
    ctx.save();
    ctx.strokeStyle = pal.fg;
    ctx.globalAlpha = 0.55;
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    for (const p of breaks) {
      const x = Math.round(GUTTER_L + p * cellW) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, PAD_T);
      ctx.lineTo(x, PAD_T + nLayers * ROW_H);
      ctx.stroke();
    }
    ctx.restore();

    // Layer labels: every 5th plus the last, or the grid legend is unreadable.
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    for (let k = 0; k < nLayers; k++) {
      const isActive = k === activeLayer;
      if (!isActive && k % 5 !== 0 && k !== nLayers - 1) continue;
      ctx.fillStyle = isActive ? pal.fg : pal.muted;
      ctx.textAlign = "right";
      ctx.fillText(layerNames[k], GUTTER_L - 6, PAD_T + k * ROW_H + ROW_H / 2);
    }

    // Active layer row.
    ctx.strokeStyle = pal.fg;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(
      GUTTER_L - 1.5,
      PAD_T + activeLayer * ROW_H - 0.5,
      cssW - GUTTER_L - PAD_R + 3,
      ROW_H
    );

    // Pinned column.
    if (pinned >= 0) {
      ctx.strokeStyle = pal.pin;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        GUTTER_L + pinned * cellW - 1,
        PAD_T - 1.5,
        Math.max(2, cellW) + 2,
        nLayers * ROW_H + 2
      );
    }

    // Time axis, keyed off the ORDINAL (never the absolute position -- see the
    // module note): ordinal i sits at i * seconds_per_position.
    const axisY = PAD_T + nLayers * ROW_H;
    ctx.strokeStyle = pal.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(GUTTER_L, axisY + 0.5);
    ctx.lineTo(cssW - PAD_R, axisY + 0.5);
    ctx.stroke();
    ctx.fillStyle = pal.muted;
    ctx.textAlign = "center";
    const total = nPos * spp;
    const step = total > 20 ? 5 : total > 6 ? 2 : 1;
    for (let t = 0; t <= total + 1e-9; t += step) {
      const x = GUTTER_L + (t / spp) * cellW;
      if (x > cssW - PAD_R) break;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, axisY);
      ctx.lineTo(Math.round(x) + 0.5, axisY + 4);
      ctx.stroke();
      ctx.fillText(`${t.toFixed(0)}s`, x, axisY + 12);
    }
    ctx.textAlign = "right";
    ctx.fillText(`${total.toFixed(2)}s`, cssW - PAD_R, axisY + 12);

    canvas.setAttribute(
      "aria-label",
      `${nLayers} layers by ${nPos} audio positions; active ${layerNames[activeLayer]}, ` +
        `${stats[activeLayer].junk} of ${nPos} junk`
    );
  }

  function schedule() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      draw();
    });
  }

  // ---------------------------------------------------------- DOM sync ----
  function renderReadout() {
    const s = stats[activeLayer];
    const extra = s.undecodable ? ` · ${s.undecodable} undecodable` : "";
    readout.textContent =
      `${layerNames[activeLayer]} · ${s.junk}/${nPos} junk · ${s.unique} unique` +
      `${extra} · modal '${escapeToken(s.modal)}' ×${s.modalCount} · ` +
      `${s.settled}/${nPos} already equal the final token`;
  }

  function renderChips() {
    chipsHead.textContent =
      `${layerNames[activeLayer]} — all ${nPos} probe tokens, in time order`;
    for (let p = 0; p < nPos; p++) {
      const slot = at(p, activeLayer);
      const cls = classOf(slot);
      const chip = chipEls[p];
      chip.textContent = chipLabel(vocab[slot]);
      chip.className =
        "pg-chip" +
        (cls === JUNK ? " is-junk" : cls === UNDECODABLE ? " is-undecodable" : "") +
        (p === pinned ? " is-pinned" : "");
      chip.title = describe(p, activeLayer);
    }
  }

  function renderPin() {
    if (pinned < 0) {
      pinPanel.hidden = true;
      return;
    }
    pinPanel.hidden = false;
    pinHead.textContent =
      `pos ${positions[pinned]} · audio token ${pinned + 1}/${nPos} · ` +
      `t≈${timeLabel(pinned)} — its ${nLayers}-layer trajectory`;
    for (let k = 0; k < nLayers; k++) {
      const slot = at(pinned, k);
      const cls = classOf(slot);
      const row = pinRows[k];
      row.bar.className =
        "pg-pin-bar" +
        (cls === JUNK ? " is-junk" : cls === UNDECODABLE ? " is-undecodable" : "") +
        (slot === finalOf[pinned] ? " is-settled" : "");
      row.token.textContent = chipLabel(vocab[slot]);
      row.token.className =
        "pg-pin-token" + (cls === JUNK ? " is-junk" : cls === UNDECODABLE ? " is-undecodable" : "");
      row.row.classList.toggle("is-active", k === activeLayer);
    }
  }

  function setLayer(k) {
    const next = Math.max(0, Math.min(nLayers - 1, k));
    if (next === activeLayer) return;
    activeLayer = next;
    renderReadout();
    renderChips();
    renderPin();
    schedule();
  }

  function setPinned(p) {
    pinned = p;
    renderChips();
    renderPin();
    schedule();
  }

  // ------------------------------------------------------ interaction ----
  function hit(event) {
    const rect = canvas.getBoundingClientRect();
    // `layout()` clamps the drawing space to MIN_W, but the stylesheet's
    // `max-width:100%` still applies — so in a container narrower than MIN_W the
    // canvas renders smaller than its coordinate space. Map client px back into
    // drawing px, or every pointer target drifts right in a narrow pane.
    const sx = cssW / (rect.width || cssW);
    const sy = cssH / (rect.height || cssH);
    const x = (event.clientX - rect.left) * sx;
    const y = (event.clientY - rect.top) * sy;
    const p = Math.floor((x - GUTTER_L) / cellW);
    const k = Math.floor((y - PAD_T) / ROW_H);
    if (p < 0 || p >= nPos || k < 0 || k >= nLayers) return null;
    return { p, k, x, y };
  }

  let dragging = false;
  let dragStart = null;

  function onPointerDown(event) {
    const spot = hit(event);
    if (!spot) return;
    dragging = true;
    dragStart = spot;
    if (canvas.setPointerCapture) canvas.setPointerCapture(event.pointerId);
    setLayer(spot.k);
    event.preventDefault();
  }

  function onPointerMove(event) {
    const spot = hit(event);
    if (dragging && spot) setLayer(spot.k);
    if (!spot) {
      tip.hidden = true;
      return;
    }
    tip.hidden = false;
    tip.textContent = `${layerNames[spot.k]}\n${describe(spot.p, spot.k)}`;
    // Flip the tooltip to the left near the right edge so it never clips.
    const wrapW = canvasWrap.clientWidth || cssW;
    const flip = spot.x > wrapW - 240;
    tip.style.left = flip ? "" : Math.round(spot.x + 12) + "px";
    tip.style.right = flip ? Math.round(wrapW - spot.x + 12) + "px" : "";
    tip.style.top = Math.round(Math.min(spot.y + 14, cssH - 44)) + "px";
  }

  function onPointerUp(event) {
    const spot = hit(event) || dragStart;
    // A tap pins the column; a drag was a layer scrub and must not also pin
    // whatever it ended on. The slop is in *pixels*, not cells: at 248 columns
    // a cell is ~3 px wide, so a cell-based test would read an ordinary hand
    // tremor as a drag and the column would never pin.
    const travel = dragStart
      ? Math.hypot(spot.x - dragStart.x, spot.y - dragStart.y)
      : Infinity;
    if (dragStart && spot && travel <= 6) {
      setPinned(spot.p === pinned ? -1 : spot.p);
    }
    dragging = false;
    dragStart = null;
    if (canvas.releasePointerCapture && event.pointerId != null) {
      try {
        canvas.releasePointerCapture(event.pointerId);
      } catch (e) {
        /* capture may already be gone */
      }
    }
  }

  function onPointerLeave() {
    tip.hidden = true;
  }

  function onKeyDown(event) {
    if (event.key === "ArrowUp") setLayer(activeLayer - 1);
    else if (event.key === "ArrowDown") setLayer(activeLayer + 1);
    else if (event.key === "Home") setLayer(0);
    else if (event.key === "End") setLayer(nLayers - 1);
    else return;
    event.preventDefault();
  }

  canvas.tabIndex = 0;
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("pointerleave", onPointerLeave);
  canvas.addEventListener("keydown", onKeyDown);

  function onToggle() {
    // Default (unchecked) greys and strikes the junk chips: the degenerate
    // reading is what a student should meet first. The toggle turns the greying
    // OFF so they can see the raw strip and argue with the classification.
    el.classList.toggle("pg-plain", toggle.checked);
  }
  toggle.addEventListener("change", onToggle);

  function onChipClick(event) {
    const chip = event.target.closest ? event.target.closest(".pg-chip") : null;
    if (!chip || chip.dataset.pos == null) return;
    const p = Number(chip.dataset.pos);
    setPinned(p === pinned ? -1 : p);
    if (!reduceMotion && pinPanel.scrollIntoView) {
      pinPanel.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
  chips.addEventListener("click", onChipClick);

  let observer = null;
  if (typeof ResizeObserver === "function") {
    observer = new ResizeObserver(schedule);
    observer.observe(canvasWrap);
  } else {
    window.addEventListener("resize", schedule);
  }

  // marimo flips its theme by mutating attributes on <html>; redraw so the
  // canvas colours follow the CSS custom properties instead of freezing.
  let themeObserver = null;
  if (typeof MutationObserver === "function") {
    themeObserver = new MutationObserver(schedule);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme", "style"],
    });
  }
  const media =
    typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: dark)") : null;
  if (media && media.addEventListener) media.addEventListener("change", schedule);

  renderReadout();
  renderChips();
  renderPin();
  draw();

  return () => {
    if (frame) cancelAnimationFrame(frame);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("pointercancel", onPointerUp);
    canvas.removeEventListener("pointerleave", onPointerLeave);
    canvas.removeEventListener("keydown", onKeyDown);
    toggle.removeEventListener("change", onToggle);
    chips.removeEventListener("click", onChipClick);
    if (observer) observer.disconnect();
    else window.removeEventListener("resize", schedule);
    if (themeObserver) themeObserver.disconnect();
    if (media && media.removeEventListener) media.removeEventListener("change", schedule);
  };
}

export function render({ model, el }) {
  el.classList.add("ctp-probe-grid");
  let teardown = null;

  const rebuild = () => {
    if (teardown) {
      try {
        teardown();
      } catch (e) {
        /* a failed teardown must not block the rebuild */
      }
      teardown = null;
    }
    el.textContent = "";
    try {
      teardown = build(model, el);
    } catch (err) {
      showError(el, err);
    }
  };

  // Listening for py->js updates is fine; writing back is what we never do.
  const keys = [
    "layer_names",
    "positions",
    "vocab",
    "vocab_class",
    "token_index",
    "seconds_per_position",
    "junk_definition",
    "caveat",
  ];

  try {
    rebuild();
    keys.forEach((key) => model.on("change:" + key, rebuild));
  } catch (err) {
    showError(el, err);
  }

  return () => {
    try {
      keys.forEach((key) => model.off("change:" + key, rebuild));
      if (teardown) teardown();
    } catch (e) {
      /* nothing useful to do while unmounting */
    }
  };
}
