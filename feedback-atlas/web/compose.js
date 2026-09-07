/* The submission form (PRD 5.2, 5.3).
 *
 * Exactly one thing can block a submission: empty text. Everything else falls
 * back to a default, because this is used *during* a presentation and a form
 * that argues with someone mid-talk simply gets abandoned. The server enforces
 * the same rule; this is the copy of it that keeps the round trip off the
 * critical path.
 *
 * The AI/사람 toggle is available to everyone regardless of role, per 5.2 -- an
 * auditor adding an AI reading is the point, not an exception.
 */

export function mountCompose({ state, send, atlas }) {
  const $ = (id) => document.getElementById(id);
  const form = $("compose-form");
  const text = $("c-text");
  const target = $("c-target");
  const week = $("c-week");
  const seg = $("c-source");
  const submit = $("c-submit");
  const hint = $("c-hint");
  let source = "ai";                       // PRD 5.3: the default is AI
  let pending = null;

  seg.addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    source = button.dataset.value;
    for (const b of seg.children)
      b.setAttribute("aria-pressed", String(b === button));
  });

  document.addEventListener("atlas:hello", () => {
    target.replaceChildren();
    for (const t of state.targets) {
      const option = document.createElement("option");
      option.value = t.id;
      option.textContent = t.display_name;
      target.appendChild(option);
    }
    week.replaceChildren();
    for (const w of [1, 2, 3, 4]) {
      const option = document.createElement("option");
      option.value = String(w);
      option.textContent = `${w}주차`;
      option.selected = w === state.defaults.week;
      week.appendChild(option);
    }
    source = state.defaults.source;
    for (const b of seg.children)
      b.setAttribute("aria-pressed", String(b.dataset.value === source));
    text.maxLength = state.defaults.maxChars;
    // The admin channel observes; it does not write. Hide the whole section, not
    // just the form -- leaving the wrapper visible parks an empty 330px column
    // with a heading over nothing between the map and the admin tools.
    const admin = state.channel === "admin";
    document.getElementById("compose").hidden = admin;
    document.getElementById("tab-compose").hidden = admin;
  });

  document.addEventListener("atlas:conn", (e) => {
    const online = e.detail;
    submit.disabled = !online || pending != null;
    // Says why it is disabled. A submission silently swallowed while a phone
    // reconnects is the worst outcome available here.
    submit.textContent = online ? (pending ? "보내는 중…" : "보내기") : "재연결 중";
  });

  document.addEventListener("atlas:ack", (e) => {
    if (pending && e.detail.nonce === pending.nonce) {
      pending = null;
      text.value = "";
      submit.disabled = !state.connected;
      submit.textContent = "보내기";
      hint.textContent = "보냈습니다. 지도에 곧 나타납니다.";
      text.focus();
    }
  });

  document.addEventListener("atlas:error", (e) => {
    if (!pending || e.detail.fatal) return;
    pending = null;
    submit.disabled = !state.connected;
    submit.textContent = "보내기";
    hint.textContent = `${e.detail.message} ${e.detail.hint || ""}`.trim();
    hint.style.color = "var(--bad)";
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const body = text.value.trim();
    if (!body) {
      hint.textContent = "의견을 입력해 주세요.";
      hint.style.color = "var(--bad)";
      text.focus();
      return;
    }
    hint.style.color = "";
    const nonce = `c-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    pending = { nonce };
    submit.disabled = true;
    submit.textContent = "보내는 중…";
    // No reviewer_id. Authorship is the connection's, and the server would
    // ignore the field anyway -- sending one would only imply it mattered.
    const ok = send({
      t: "submit", nonce,
      target_id: target.value,
      text: body,
      source,
      week: Number(week.value),
    });
    if (!ok) {
      pending = null;
      submit.disabled = false;
      submit.textContent = "재연결 중";
      hint.textContent = "연결이 끊겼습니다. 잠시 후 다시 시도해 주세요.";
      hint.style.color = "var(--bad)";
    }
  });

  // Ctrl/Cmd+Enter sends, so a phone keyboard's newline still works.
  text.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") form.requestSubmit();
  });
}
