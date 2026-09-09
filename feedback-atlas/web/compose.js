/* The submission form (PRD 5.2, 5.3).
 *
 * Week assignment and the acceptance window come from the server calendar.
 * Draft text stays in place when the period closes or a send fails.
 *
 * The AI/사람 toggle is available to everyone regardless of role, per 5.2 -- an
 * observer adding an AI reading is the point, not an exception.
 */

export function mountCompose({ state, send, atlas }) {
  const $ = (id) => document.getElementById(id);
  const form = $("compose-form");
  const text = $("c-text");
  const target = $("c-target");
  const weekStatus = $("c-week-status");
  const seg = $("c-source");
  const submit = $("c-submit");
  const hint = $("c-hint");
  let source = "ai";                       // PRD 5.3: the default is AI
  let pending = null;
  let initialized = false;
  let targetNeedsChoice = false;
  let composeChoice = null;

  function updatePlaceholder() {
    text.placeholder = source === "ai"
      ? "AI가 생성한 피드백을 붙여넣어 주세요."
      : "느낀 그대로 적어 주세요.";
  }

  function updateAvailability() {
    const accepting = state.schedule?.accepting !== false;
    submit.disabled = !state.connected || pending != null || !accepting;
    submit.textContent = !state.connected ? "재연결 중"
      : pending ? "보내는 중…" : accepting ? "보내기" : "접수 기간이 아닙니다";
    const schedule = state.schedule;
    if (!schedule) { weekStatus.textContent = ""; return; }
    if (schedule.demo) {
      weekStatus.textContent = `Demo · ${schedule.current_week}주차로 연습합니다.`;
    } else if (schedule.current_week) {
      const period = schedule.periods.find(p => p.week === schedule.current_week);
      weekStatus.textContent = `${schedule.current_week}주차 · ${period.start} ~ ${period.end} (한국 시간, 자동 분류)`;
    } else {
      weekStatus.textContent = "현재는 수업 의견 접수 기간이 아닙니다. 작성 중인 의견은 유지됩니다.";
    }
  }

  updatePlaceholder();
  document.addEventListener("atlas:schedule", updateAvailability);

  seg.addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    source = button.dataset.value;
    for (const b of seg.children)
      b.setAttribute("aria-pressed", String(b === button));
    updatePlaceholder();
  });

  target.addEventListener("change", () => {
    if (target.value) targetNeedsChoice = false;
  });

  function rebuildTargets() {
    const wantedTarget = initialized ? target.value : "";
    const wantedSource = initialized ? source : state.defaults.source;
    let targetFound = false;
    target.replaceChildren();
    for (const t of state.targets) {
      const option = document.createElement("option");
      option.value = t.id;
      option.textContent = t.display_name;
      if (initialized && t.id === wantedTarget) {
        option.selected = true;
        targetFound = true;
      }
      target.appendChild(option);
    }
    targetNeedsChoice = targetNeedsChoice || (initialized && wantedTarget && !targetFound);
    if (targetNeedsChoice) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "대상을 다시 선택해 주세요";
      option.selected = true;
      option.disabled = true;
      target.prepend(option);
      hint.textContent = "대상이 바뀌었습니다. 다시 선택해 주세요.";
      hint.style.color = "var(--bad)";
    }
    source = wantedSource;
    for (const b of seg.children)
      b.setAttribute("aria-pressed", String(b.dataset.value === source));
    text.maxLength = state.defaults.maxChars;
    updatePlaceholder();
    updateAvailability();
    initialized = true;
  }

  function applyChannelVisibility() {
    // The admin channel observes; it does not write. Hide the whole section, not
    // just the form -- leaving the wrapper visible parks an empty 330px column
    // with a heading over nothing between the map and the admin tools.
    const admin = state.channel === "admin";
    const panel = document.getElementById("compose");
    const button = document.getElementById("tab-compose");
    button.hidden = admin;
    if (admin) {
      panel.hidden = true;
      return;
    }
    if (composeChoice !== null) {
      panel.hidden = !composeChoice;
      button.setAttribute("aria-pressed", String(composeChoice));
      return;
    }
    panel.hidden = false;
    button.setAttribute("aria-pressed", "true");
  }

  document.getElementById("tab-compose").addEventListener("click", () => {
    if (state.channel === "admin") return;
    composeChoice = document.getElementById("tab-compose").getAttribute("aria-pressed") === "true";
  });

  document.addEventListener("atlas:hello", () => {
    if (pending && pending.owner !== state.submissionOwner) {
      pending = null;
      updateAvailability();
      hint.textContent = "로그인한 사용자가 바뀌어 이전 전송은 자동 재시도하지 않습니다. 초안은 유지됩니다.";
      hint.style.color = "var(--bad)";
    }
    rebuildTargets();
    applyChannelVisibility();
  });

  document.addEventListener("atlas:roster", () => {
    rebuildTargets();
  });

  document.addEventListener("atlas:conn", (e) => {
    const online = e.detail;
    // Says why it is disabled. A submission silently swallowed while a phone
    // reconnects is the worst outcome available here.
    updateAvailability();
    if (online && pending && pending.owner === state.submissionOwner) send(pending.frame);
  });

  document.addEventListener("atlas:ack", (e) => {
    if (pending && e.detail.nonce === pending.nonce) {
      const sentDraft = pending.draft;
      pending = null;
      if (text.value === sentDraft) text.value = "";
      updateAvailability();
      hint.textContent = "보냈습니다. 지도에 곧 나타납니다.";
      text.focus();
    }
  });

  document.addEventListener("atlas:error", (e) => {
    if (!pending || e.detail.fatal) return;
    pending = null;
    updateAvailability();
    hint.textContent = `${e.detail.message} ${e.detail.hint || ""}`.trim();
    hint.style.color = "var(--bad)";
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (pending) return;
    if (state.schedule?.accepting === false) {
      hint.textContent = "현재는 수업 의견 접수 기간이 아닙니다.";
      hint.style.color = "var(--bad)";
      return;
    }
    const body = text.value.trim();
    if (!body) {
      hint.textContent = "의견을 입력해 주세요.";
      hint.style.color = "var(--bad)";
      text.focus();
      return;
    }
    if (!target.value) {
      hint.textContent = "대상을 다시 선택해 주세요.";
      hint.style.color = "var(--bad)";
      target.focus();
      return;
    }
    hint.style.color = "";
    const nonce = `c-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    // No reviewer_id. Authorship is the connection's, and the server would
    // ignore the field anyway -- sending one would only imply it mattered.
    const frame = {
      t: "submit", nonce,
      target_id: target.value,
      text: body,
      source,
    };
    pending = { nonce, frame, draft: text.value, owner: state.submissionOwner };
    submit.disabled = true;
    submit.textContent = state.connected ? "보내는 중…" : "재연결 중";
    const ok = send(frame);
    if (!ok) {
      submit.textContent = "재연결 중";
      hint.textContent = "연결이 끊겼습니다. 잠시 후 다시 보내겠습니다.";
      hint.style.color = "var(--bad)";
    }
  });

  // Ctrl/Cmd+Enter sends, so a phone keyboard's newline still works.
  text.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !pending)
      form.requestSubmit();
  });
}
