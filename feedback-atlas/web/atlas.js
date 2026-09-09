/* The SVG scatter.
 *
 * Written rather than borrowed. Embedding Atlas's EmbeddingView encodes exactly
 * one channel -- category, as colour -- and PRD 5.4 needs two at once: colour for
 * the target student, shape for AI versus human. Its 0.24 line is also WebGPU-only,
 * which on a room's worth of assorted laptops means a blank canvas for some of
 * them. At a few hundred points SVG costs nothing and buys hit-testing, per-point
 * shapes and enter transitions for free.
 *
 * The camera is owned here and never derived from the data. Embedding Atlas
 * recomputes its viewport whenever the data identity changes; doing that in a
 * live classroom would move the map under a reader's cursor every time somebody
 * else submitted.
 */

// d3 category10, as Embedding Atlas uses.
export const CATEGORY10 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];
const NS = "http://www.w3.org/2000/svg";
const R = 5.2;                    // point radius in CSS px
const DIM_OPACITY = 0.06;         // a week that is filtered out
const OTHER = "var(--other)";

function el(name, attrs) {
  const node = document.createElementNS(NS, name);
  for (const k in attrs) if (attrs[k] != null) node.setAttribute(k, attrs[k]);
  return node;
}

/* Shape encodes source. Circle for a person, diamond for a machine -- round is
 * organic, angular is synthetic, which is the mnemonic people keep without a
 * legend.
 *
 * Diamond rather than triangle for a measurable reason: a triangle's optical
 * centre sits below its centroid, so a triangle reads as offset from the very
 * coordinate it is meant to mark. In a plot whose entire message is position,
 * that is a distortion. A diamond's visual centre is its centre.
 *
 * Area-matched so neither shape reads as larger or more important: a diamond of
 * half-diagonal a has area 2a^2, so a = r*sqrt(pi/2) matches a circle of radius r.
 */
const DIAMOND_K = Math.sqrt(Math.PI / 2);   // ~1.2533

function markPath(cx, cy, r) {
  const a = r * DIAMOND_K;
  return `M${cx} ${cy - a}L${cx + a} ${cy}L${cx} ${cy + a}L${cx - a} ${cy}Z`;
}

/* Ray casting. A lasso is an arbitrary closed polygon, so "inside" is a crossing
 * count, not a bounding box. */
function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi)
      inside = !inside;
  }
  return inside;
}

export class Atlas {
  constructor(svg, opts = {}) {
    this.svg = svg;
    this.onHover = opts.onHover || (() => {});
    this.onCameraChange = opts.onCameraChange || (() => {});
    this.camera = { tx: 0, ty: 0, scale: 1 };
    this.points = new Map();          // id -> point record from the server
    this.nodes = new Map();           // id -> SVG element
    this.colorOf = () => OTHER;
    this.labelOf = () => "";
    this.visible = () => true;
    this.dimmed = () => false;
    this.fitted = false;

    this.onSelect = opts.onSelect || (() => {});
    this.onPick = opts.onPick || (() => {});
    this.mode = "pan";                 // "pan" | "marquee" | "lasso"
    this.selected = new Set();         // ids, admin-only
    this.matches = null;               // ids matching the search, or null

    this.gLabels = el("g", { "pointer-events": "none" });
    this.gPoints = el("g");
    this.gOverlay = el("g", { "pointer-events": "none" });
    svg.append(this.gPoints, this.gLabels, this.gOverlay);
    this._installCamera();
    this._ro = new ResizeObserver(() => this.render());
    this._ro.observe(svg);
  }

  size() {
    const r = this.svg.getBoundingClientRect();
    return { w: Math.max(1, r.width), h: Math.max(1, r.height) };
  }

  toScreen(x, y) {
    const { w, h } = this.size();
    return [w / 2 + (x + this.camera.tx) * this.camera.scale,
            h / 2 + (y + this.camera.ty) * this.camera.scale];
  }

  /* Fit every point with padding. Called on first data and on the fit button --
   * never automatically afterwards. */
  fit() {
    const pts = [...this.points.values()];
    if (!pts.length) return;
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const { w, h } = this.size();
    const pad = 56;
    const spanX = x1 - x0, spanY = y1 - y0;
    if (spanX < 1e-6 && spanY < 1e-6) {
      // Every point coincident -- one opinion, or a corpus of identical text.
      // There is no extent to fit, so pick a readable scale instead of dividing
      // by nothing and zooming to infinity.
      this.camera.scale = 120;
    } else {
      // No artificial ceiling. A cap here fires on ordinary data, not just
      // degenerate data: with a cloud spanning ~0.7 units in a 770px box the
      // honest scale is around 1100, so a cap of 400 left the map filling a
      // third of the canvas and looking empty. The degenerate case is handled
      // above, which is the only thing the cap was ever protecting against.
      this.camera.scale = Math.min(
        (w - pad * 2) / Math.max(spanX, 1e-6),
        (h - pad * 2) / Math.max(spanY, 1e-6));
    }
    this.camera.tx = -(x0 + x1) / 2;
    this.camera.ty = -(y0 + y1) / 2;
    this.fitted = true;
    this.render();
    this.onCameraChange(this.camera);
  }

  setPoints(list) {
    for (const p of list) this.points.set(p.id, p);
  }

  upsert(list, { entering = false } = {}) {
    for (const p of list) this.points.set(p.id, p);
    if (entering) this.markEntering(list.map((p) => p.id));
  }

  markEntering(ids) {
    this._entering = this._entering || new Set();
    for (const id of ids) this._entering.add(id);
  }

  move(list) {
    for (const m of list) {
      const p = this.points.get(m.id);
      if (p) { p.x = m.x; p.y = m.y; }
    }
  }

  clear() {
    this.points.clear();
    this.nodes.forEach(n => n.remove());
    this.nodes.clear();
  }

  render() {
    const seen = new Set();
    for (const [id, p] of this.points) {
      seen.add(id);
      const [cx, cy] = this.toScreen(p.x, p.y);
      const isAi = p.source === "ai";
      const vis = this.visible(p);
      const dim = vis && this.dimmed(p);
      const opacity = !vis ? DIM_OPACITY : (dim ? 0.16 : 0.72);
      const fill = dim ? OTHER : this.colorOf(p);

      let node = this.nodes.get(id);
      if (!node || node.tagName !== (isAi ? "path" : "circle")) {
        node?.remove();
        node = el(isAi ? "path" : "circle", {});
        node.dataset.id = id;
        this.gPoints.appendChild(node);
        this.nodes.set(id, node);
        if (this._entering?.has(id)) {
          // The live moment the whole tool is built around: a new opinion
          // arriving mid-presentation. Embedding Atlas has no enter transition
          // at all, which is one of the things writing the scatter buys.
          node.classList.add("entering");
          this._entering.delete(id);
          this._halo(cx, cy);
        }
      }
      if (isAi) node.setAttribute("d", markPath(cx, cy, R));
      else { node.setAttribute("cx", cx.toFixed(1)); node.setAttribute("cy", cy.toFixed(1));
             node.setAttribute("r", R); }
      node.setAttribute("fill", fill);
      node.setAttribute("opacity", opacity);
      node.style.pointerEvents = vis ? "auto" : "none";
      // Their affordance exactly: a hollow ring at stroke-width 2, filled with
      // the category colour, so a selected point still reads as its category.
      if (this.selected.has(id)) {
        node.setAttribute("stroke", "var(--fg)");
        node.setAttribute("stroke-width", "2");
        node.setAttribute("opacity", !vis ? DIM_OPACITY : (dim ? 0.16 : 0.95));
      } else {
        node.removeAttribute("stroke");
        node.removeAttribute("stroke-width");
      }
    }
    for (const [id, node] of this.nodes) {
      if (!seen.has(id)) { node.remove(); this.nodes.delete(id); }
    }
    this._renderLabels();
  }

  setMode(mode) {
    this.mode = mode;
    this.svg.style.cursor = mode === "pan" ? "grab" : "crosshair";
  }

  setSelection(ids) {
    this.selected = new Set(ids || []);
    this.render();
  }

  setMatches(ids) {
    this.matches = ids ? new Set(ids) : null;
    this.render();
  }

  /* Which visible points fall inside a screen-space shape. */
  _hits(shape) {
    const out = [];
    for (const p of this.points.values()) {
      if (!this.visible(p) || this.dimmed(p)) continue;
      const [cx, cy] = this.toScreen(p.x, p.y);
      if (shape.kind === "rect") {
        if (cx >= shape.x0 && cx <= shape.x1 && cy >= shape.y0 && cy <= shape.y1)
          out.push(p);
      } else if (pointInPolygon(cx, cy, shape.points)) {
        out.push(p);
      }
    }
    return out;
  }

  _halo(cx, cy) {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const ring = el("circle", { cx, cy, r: 4, fill: "none",
                                stroke: "var(--accent)", "stroke-width": 1.5 });
    ring.style.pointerEvents = "none";
    const grow = el("animate", { attributeName: "r", from: 4, to: 34, dur: "0.7s", fill: "freeze" });
    const fade = el("animate", { attributeName: "opacity", from: 0.6, to: 0, dur: "0.7s", fill: "freeze" });
    ring.append(grow, fade);
    this.gPoints.appendChild(ring);
    setTimeout(() => ring.remove(), 800);
  }

  /* Halo'd map labels at each target's sub-cluster centroid.
   *
   * paint-order:stroke with a round-joined 4px stroke is what makes text legible
   * over a point cloud -- four attributes, and the single most transferable idea
   * in Embedding Atlas's rendering. Collisions drop the lower-priority label, so
   * labels thin out rather than overlapping, the way a real map behaves.
   */
  _renderLabels() {
    this.gLabels.replaceChildren();
    const groups = new Map();
    for (const p of this.points.values()) {
      if (!this.visible(p) || this.dimmed(p)) continue;
      const key = `${p.target_id}|${p.source}`;
      const g = groups.get(key) || { pts: [], target: p.target_id };
      g.pts.push(p);
      groups.set(key, g);
    }

    const placed = [];
    const candidates = [...groups.values()]
      .filter((g) => g.pts.length >= 3)
      .sort((a, b) => b.pts.length - a.pts.length);

    for (const g of candidates) {
      const n = g.pts.length;
      const mx = g.pts.reduce((s, p) => s + p.x, 0) / n;
      const my = g.pts.reduce((s, p) => s + p.y, 0) / n;

      /* Anchor on the medoid -- the real point nearest the centroid -- not on the
       * centroid itself. A target's opinions are frequently bimodal (the AI
       * cluster and the human cluster sit apart, which is the finding this map
       * exists to show), and the mean of two lobes lands in the gap between them.
       * A name floating over empty space is worse than no name: it labels
       * nothing, and it invites the reader to believe there is something there.
       */
      let anchor = g.pts[0], best = Infinity, spread = 0;
      for (const p of g.pts) {
        const d = (p.x - mx) ** 2 + (p.y - my) ** 2;
        spread += Math.sqrt(d);
        if (d < best) { best = d; anchor = p; }
      }
      // Too diffuse to be a cluster at all: naming it would assert a grouping
      // the geometry does not support.
      const meanRadius = (spread / n) * this.camera.scale;
      if (meanRadius > 220) continue;

      const [cx, cy] = this.toScreen(anchor.x, anchor.y);
      if (placed.some((q) => Math.abs(q.x - cx) < 108 && Math.abs(q.y - cy) < 22)) continue;
      placed.push({ x: cx, y: cy });

      const text = el("text", {
        x: cx.toFixed(1), y: (cy - 18).toFixed(1),
        "text-anchor": "middle", "dominant-baseline": "middle",
        "font-size": 11, "font-weight": 500, "font-family": "var(--sans)",
        fill: "var(--label)", stroke: "var(--halo)", "stroke-width": 4,
        "stroke-linejoin": "round", "stroke-linecap": "round",
        "paint-order": "stroke", opacity: 0.8,
      });
      text.textContent = this.labelOf(g.target);
      this.gLabels.appendChild(text);
    }
  }

  /* A cartographic scale bar: pick a round number near 60px. */
  scaleBar() {
    const target = 60 / this.camera.scale;
    const mag = Math.pow(10, Math.floor(Math.log10(target)));
    const step = [1, 2, 5, 10].find(m => m * mag >= target) * mag;
    return { px: Math.round(step * this.camera.scale), label: String(+step.toPrecision(2)) };
  }

  _installCamera() {
    const svg = this.svg;
    let dragging = false, lastX = 0, lastY = 0, moved = 0;

    let path = null, startX = 0, startY = 0;
    const local = (e) => {
      const r = svg.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };

    svg.addEventListener("pointerdown", (e) => {
      dragging = true; moved = 0; lastX = e.clientX; lastY = e.clientY;
      svg.setPointerCapture(e.pointerId);
      if (this.mode === "pan") { svg.classList.add("dragging"); return; }
      const [x, y] = local(e);
      startX = x; startY = y;
      path = this.mode === "lasso" ? [[x, y]] : null;
      this._drawOverlay(this.mode === "lasso"
        ? { kind: "lasso", points: [[x, y]] }
        : { kind: "rect", x0: x, y0: y, x1: x, y1: y });
    });

    svg.addEventListener("pointermove", (e) => {
      if (!dragging) { this._hover(e); return; }
      if (this.mode === "pan") {
        const dx = e.clientX - lastX, dy = e.clientY - lastY;
        moved += Math.abs(dx) + Math.abs(dy);
        lastX = e.clientX; lastY = e.clientY;
        this.camera.tx += dx / this.camera.scale;
        this.camera.ty += dy / this.camera.scale;
        this.render();
        return;
      }
      const [x, y] = local(e);
      moved += Math.abs(x - startX) + Math.abs(y - startY);
      if (this.mode === "lasso") {
        path.push([x, y]);
        this._drawOverlay({ kind: "lasso", points: path });
      } else {
        this._drawOverlay({ kind: "rect", x0: Math.min(startX, x), y0: Math.min(startY, y),
                            x1: Math.max(startX, x), y1: Math.max(startY, y) });
      }
    });

    const end = (e) => {
      if (!dragging) return;
      dragging = false; svg.classList.remove("dragging");
      if (e.pointerId != null && svg.hasPointerCapture?.(e.pointerId))
        svg.releasePointerCapture(e.pointerId);
      if (this.mode === "pan") {
        // A click that did not drag is a pick, not a pan.
        if (moved < 4) {
          const hit = this._nearest(...local(e));
          this.onPick(hit ? hit.point : null);
        }
        return;
      }
      const [x, y] = local(e);
      const shape = this.mode === "lasso"
        ? { kind: "lasso", points: path || [] }
        : { kind: "rect", x0: Math.min(startX, x), y0: Math.min(startY, y),
            x1: Math.max(startX, x), y1: Math.max(startY, y) };
      this.gOverlay.replaceChildren();
      path = null;
      // A stray click in selection mode clears rather than selecting nothing in
      // particular, which is what a reader expects from an empty drag.
      const hits = (shape.kind === "rect" && (shape.x1 - shape.x0) < 3
                    && (shape.y1 - shape.y0) < 3) ? [] : this._hits(shape);
      this.onSelect(hits);
    };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
    svg.addEventListener("pointerleave", () => this.onHover(null));

    svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = svg.getBoundingClientRect();
      const mx = e.clientX - rect.left - rect.width / 2;
      const my = e.clientY - rect.top - rect.height / 2;
      const before = this.camera.scale;
      const factor = Math.exp(-e.deltaY * 0.0015);
      this.camera.scale = Math.min(4000, Math.max(4, before * factor));
      // Keep the point under the cursor under the cursor.
      this.camera.tx += mx / this.camera.scale - mx / before;
      this.camera.ty += my / this.camera.scale - my / before;
      this.render();
      this.onCameraChange(this.camera);
    }, { passive: false });
  }

  _hover(e) {
    const rect = this.svg.getBoundingClientRect();
    this.onHover(this._nearest(e.clientX - rect.left, e.clientY - rect.top));
  }

  _nearest(px, py) {
    let best = null, bestD = (R + 6) ** 2;
    for (const p of this.points.values()) {
      if (!this.visible(p)) continue;
      const [cx, cy] = this.toScreen(p.x, p.y);
      const d = (cx - px) ** 2 + (cy - py) ** 2;
      if (d < bestD) { bestD = d; best = { point: p, cx, cy }; }
    }
    return best;
  }

  /* Marquee and lasso, styled as Embedding Atlas styles them: a slate-bordered
   * rectangle over a faint wash, and a white-stroked lasso path. */
  _drawOverlay(shape) {
    this.gOverlay.replaceChildren();
    if (shape.kind === "rect") {
      this.gOverlay.appendChild(el("rect", {
        x: shape.x0, y: shape.y0,
        width: Math.max(0, shape.x1 - shape.x0),
        height: Math.max(0, shape.y1 - shape.y0),
        fill: "rgba(128,128,128,.18)", stroke: "var(--fg)",
        "stroke-width": 1, "stroke-dasharray": "3 2",
      }));
    } else {
      const d = shape.points.map((q, i) => `${i ? "L" : "M"}${q[0]} ${q[1]}`).join("");
      this.gOverlay.appendChild(el("path", {
        d: d + "Z", fill: "rgba(128,128,128,.25)",
        stroke: "var(--fg)", "stroke-width": 1.2, "stroke-linejoin": "round",
      }));
    }
  }
}
