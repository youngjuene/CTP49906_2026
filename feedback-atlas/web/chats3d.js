// The chat map in 3D: the build's 3D cloud (x3, y3, z3) drawn on a canvas with
// perspective, and every step of a conversation drawn as a SLERP arc.
//
// The arc is spherical interpolation of the direction from the cloud's centre,
// with the distance from the centre blended linearly. Between two sentences it
// curves around the middle of the corpus instead of cutting straight through it.
// The bend is a drawing choice, not a property of the data: the build normalises
// the cloud to be centred on the origin, and that centre is all the arc refers to.
//
// No library: the site only runs its own scripts, and a few thousand points do
// not need WebGL. Points are depth-sorted and painted back to front each frame.

const TAU = Math.PI * 2;

// SLERP of a and b about the origin, radius blended linearly. t in [0, 1].
export function arc(a, b, t) {
  const ra = Math.hypot(a.x3, a.y3, a.z3), rb = Math.hypot(b.x3, b.y3, b.z3);
  if (ra < 1e-9 || rb < 1e-9) return lerp3(a, b, t);
  const ua = [a.x3 / ra, a.y3 / ra, a.z3 / ra], ub = [b.x3 / rb, b.y3 / rb, b.z3 / rb];
  const dot = Math.max(-1, Math.min(1, ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2]));
  const om = Math.acos(dot);
  // Nearly the same direction, or exactly opposite (no unique great circle):
  // a straight line is the honest fallback.
  if (om < 1e-4 || Math.PI - om < 1e-3) return lerp3(a, b, t);
  const s = Math.sin(om), wa = Math.sin((1 - t) * om) / s, wb = Math.sin(t * om) / s;
  const r = ra + (rb - ra) * t;
  return { x3: r * (wa * ua[0] + wb * ub[0]), y3: r * (wa * ua[1] + wb * ub[1]), z3: r * (wa * ua[2] + wb * ub[2]) };
}
const lerp3 = (a, b, t) => ({ x3: a.x3 + (b.x3 - a.x3) * t, y3: a.y3 + (b.y3 - a.y3) * t, z3: a.z3 + (b.z3 - a.z3) * t });

// 3x3 rotations, row-major arrays.
const mul = (A, B) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c]));
function axisAngle([x, y, z], a) {
  const c = Math.cos(a), s = Math.sin(a), t = 1 - c;
  return [t * x * x + c, t * x * y - s * z, t * x * z + s * y,
          t * x * y + s * z, t * y * y + c, t * y * z - s * x,
          t * x * z - s * y, t * y * z + s * x, t * z * z + c];
}
const START = mul(axisAngle([1, 0, 0], -0.35), axisAngle([0, 1, 0], 0.6));

export function createView3d(canvas, { onTap }) {
  const ctx = canvas.getContext("2d");
  const cam = { R: START.slice(), zoom: 1 };
  const DIST = 4.2;                 // camera distance, in cloud radii
  let size = [0, 0], focal = 1, last = null;

  function resize() {
    const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    size = [r.width, r.height];
    canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    focal = 0.3 * Math.min(r.width, r.height) * DIST;   // the cloud's ~1.5 radius fills ~90% of the short side
  }

  // World -> screen: rotate, then perspective. z grows toward the viewer.
  function project(p) {
    const R = cam.R;
    const x = R[0] * p.x3 + R[1] * p.y3 + R[2] * p.z3;
    const y = R[3] * p.x3 + R[4] * p.y3 + R[5] * p.z3;
    const z = R[6] * p.x3 + R[7] * p.y3 + R[8] * p.z3;
    const k = (focal * cam.zoom) / (DIST - z);
    return { sx: size[0] / 2 + x * k, sy: size[1] / 2 - y * k, z, k };
  }

  function strokeArcs(list, head) {
    // One path of SLERP arcs through `list`, then to `head` (the gliding cursor).
    const pts = head ? [...list, head] : list;
    if (pts.length < 2) return;
    let q = project(pts[0]);
    ctx.moveTo(q.sx, q.sy);
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1], b = pts[i];
      const ra = Math.hypot(a.x3, a.y3, a.z3), rb = Math.hypot(b.x3, b.y3, b.z3);
      const cos = (a.x3 * b.x3 + a.y3 * b.y3 + a.z3 * b.z3) / Math.max(ra * rb, 1e-9);
      const n = Math.max(2, Math.min(16, Math.ceil(Math.acos(Math.max(-1, Math.min(1, cos))) * 10)));
      for (let s = 1; s <= n; s++) { q = project(arc(a, b, s / n)); ctx.lineTo(q.sx, q.sy); }
    }
  }

  function shape(q, kind, speaker, r) {
    ctx.beginPath();
    if (kind === "status") {
      ctx.moveTo(q.sx, q.sy - r * 1.25); ctx.lineTo(q.sx + r * 1.25, q.sy);
      ctx.lineTo(q.sx, q.sy + r * 1.25); ctx.lineTo(q.sx - r * 1.25, q.sy); ctx.closePath();
    } else if (speaker === "student") ctx.rect(q.sx - r, q.sy - r, 2 * r, 2 * r);
    else ctx.arc(q.sx, q.sy, r, 0, TAU);
  }

  // `scene` is everything the page decides: which points, how each is coloured,
  // which paths are walked, where the cursor and the route head are.
  function draw(scene) {
    last = scene;
    if (!size[0]) resize();
    const css = getComputedStyle(document.documentElement);
    const v = (name) => css.getPropertyValue(name).trim();
    ctx.clearRect(0, 0, size[0], size[1]);

    const projected = scene.points.map((p) => ({ p, q: project(p) }));
    let zMin = Infinity, zMax = -Infinity;
    for (const { q } of projected) { zMin = Math.min(zMin, q.z); zMax = Math.max(zMax, q.z); }
    const depth = (z) => 0.35 + 0.65 * ((z - zMin) / Math.max(zMax - zMin, 1e-6));   // far 0.35 .. near 1

    // Paths first, points over them. Only two: the turn route and the current
    // turn. Arcs curve around the centre, so every earlier step drawn as well
    // turns the cloud into a ball of loops by mid-conversation; the route
    // already says where the conversation has been.
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    const path = (list, head, colour, width, alpha) => {
      ctx.beginPath(); strokeArcs(list, head);
      ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.globalAlpha = alpha; ctx.stroke();
    };
    if (scene.route.length) path(scene.route, scene.routeHead, v("--path-walked"), 2.5, 0.8);
    // The cursor's trail: its last few steps, oldest faintest, so the arc it is
    // travelling now reads clearly even inside a sixty-sentence answer.
    const trail = [...scene.trail, scene.cursor];
    for (let i = 1; i < trail.length; i++) {
      path([trail[i - 1], trail[i]], null, v("--accent"), 2.2, 0.95 * (i / (trail.length - 1)) ** 1.5);
    }

    // Other conversations are grey context: drawn in four depth bands, one fill
    // per band, no outline, no sort. Per-point fills for thousands of them are
    // what made a drag stutter at ten conversations.
    const card = v("--card");
    const bands = [[], [], [], []], own = [];
    let greyAlpha = 0, grey = null;
    for (const item of projected) {
      if (scene.mine(item.p)) { own.push(item); continue; }
      const st = scene.style(item.p);
      if (!st) continue;
      grey = st.colour; greyAlpha = st.alpha;
      bands[Math.min(3, Math.floor(4 * (item.q.z - zMin) / Math.max(zMax - zMin, 1e-6)))].push(item);
    }
    bands.forEach((band, i) => {
      if (!band.length) return;
      ctx.beginPath();
      // Squares: at 2-3 px they read as dots and rasterise far cheaper than arcs.
      for (const { q } of band) {
        const r = Math.max(1.2, Math.min(6, 2.6 * q.k / (focal / DIST)));
        ctx.rect(q.sx - r, q.sy - r, 2 * r, 2 * r);
      }
      ctx.globalAlpha = greyAlpha * (0.35 + 0.65 * (i + 0.5) / 4);
      ctx.fillStyle = grey; ctx.fill();
    });

    own.sort((a, b) => a.q.z - b.q.z);
    for (const { p, q } of own) {
      const st = scene.style(p);
      if (!st) continue;
      // About the flat map's 3.6 px at the cloud's centre depth, larger nearer.
      const r = Math.max(1.6, Math.min(9, 3.4 * q.k / (focal / DIST)));
      ctx.globalAlpha = st.alpha * depth(q.z);
      shape(q, p.kind, p.speaker, r * 1.05);
      if (p.kind === "status") { ctx.strokeStyle = st.colour; ctx.lineWidth = 1; ctx.stroke(); }
      else { ctx.fillStyle = st.colour; ctx.fill(); ctx.strokeStyle = card; ctx.lineWidth = 0.5; ctx.stroke(); }
    }

    // Route stops, the cursor, labels.
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const c of scene.stops) {
      const q = project(c);
      ctx.globalAlpha = depth(q.z);
      ctx.beginPath(); ctx.arc(q.sx, q.sy, 7, 0, TAU);
      ctx.fillStyle = c.now ? v("--accent") : card; ctx.fill();
      ctx.strokeStyle = c.now ? v("--accent") : v("--path-walked"); ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = c.now ? "#fff" : v("--fg"); ctx.font = "700 8.5px system-ui, sans-serif";
      ctx.fillText(String(c.turn), q.sx, q.sy + 0.5);
    }
    if (scene.cursor) {
      const q = project(scene.cursor);
      ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(q.sx, q.sy, 9, 0, TAU);
      ctx.strokeStyle = v("--accent"); ctx.lineWidth = 2.5; ctx.stroke();
      halo(scene.turnMark, q.sx + 14, q.sy - 12, v("--label") || v("--fg"), v("--halo") || card, "600 11px system-ui, sans-serif", "left");
    }
    for (const l of scene.labels) {
      const q = project(l);
      ctx.globalAlpha = 0.25 + 0.75 * depth(q.z);
      halo(l.label, q.sx, q.sy - 12, l.color, card, "700 11.5px system-ui, sans-serif", "center");
    }
    ctx.globalAlpha = 1;
  }

  function halo(text, x, y, fill, ring, font, align) {
    ctx.font = font; ctx.textAlign = align;
    ctx.lineWidth = 3.5; ctx.strokeStyle = ring; ctx.strokeText(text, x, y);
    ctx.fillStyle = fill; ctx.fillText(text, x, y);
  }

  // Turn the cloud a little so `p` comes toward the front and centre.
  function lookToward(p, rate = 0.06) {
    const R = cam.R;
    const v = [R[0] * p.x3 + R[1] * p.y3 + R[2] * p.z3, R[3] * p.x3 + R[4] * p.y3 + R[5] * p.z3,
               R[6] * p.x3 + R[7] * p.y3 + R[8] * p.z3];
    const n = Math.hypot(...v);
    if (n < 0.15) return false;                  // near the centre: nothing to turn toward
    const axis = [v[1], -v[0], 0], an = Math.hypot(...axis);   // v x (0,0,1)
    if (an < 1e-6) return false;
    const angle = Math.atan2(an, v[2]) * rate;
    if (angle < 1e-4) return false;
    cam.R = mul(axisAngle(axis.map((c) => -c / an), angle), cam.R);
    return true;
  }

  // ---- interaction: drag to rotate, wheel / pinch to zoom, tap to pick ----
  const pointers = new Map();
  let drag = null, pinch = null, moved = 0;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    moved = 0;
    if (pointers.size === 1) { drag = { x: e.clientX, y: e.clientY }; canvas.classList.add("dragging"); }
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), zoom: cam.zoom };
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2 && pinch) {
      const [a, b] = [...pointers.values()];
      cam.zoom = Math.max(0.4, Math.min(8, pinch.zoom * Math.hypot(a.x - b.x, a.y - b.y) / pinch.d));
      moved = Infinity;
    } else if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag = { x: e.clientX, y: e.clientY };
      moved += Math.hypot(dx, dy);
      cam.R = mul(axisAngle([1, 0, 0], dy * 0.008), mul(axisAngle([0, 1, 0], dx * 0.008), cam.R));
    } else return;
    if (last) draw(last);
  });
  function release(e) {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = null;
    if (pointers.size) return;
    drag = null; canvas.classList.remove("dragging");
    if (moved < 4 && e.type === "pointerup" && last) pick(e);
  }
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam.zoom = Math.max(0.4, Math.min(8, cam.zoom * Math.exp(-e.deltaY * 0.0015)));
    if (last) draw(last);
  }, { passive: false });
  canvas.addEventListener("dblclick", () => { reset(); if (last) draw(last); });

  function pick(e) {
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    // The nearest point under the finger, preferring the front one and the
    // selected conversation, like the flat map.
    let best = null, bestD = 14;
    for (const p of last.points) {
      if (!last.style(p)) continue;
      const q = project(p);
      const d = Math.hypot(q.sx - px, q.sy - py) - (last.mine(p) ? 2 : 0) - q.z;
      if (d < bestD) { best = p; bestD = d; }
    }
    if (best) onTap(best);
  }

  function reset() { cam.R = START.slice(); cam.zoom = 1; }

  return { draw, resize, lookToward, reset, redraw: () => last && draw(last) };
}
