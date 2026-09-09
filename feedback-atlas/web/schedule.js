import { apiUrl } from "./app-context.js";

const TIMEZONE = "Asia/Seoul";
const DEFAULT_PERIODS = [
  { week: 1, start: "2026-10-15", end: "2026-10-21" },
  { week: 2, start: "2026-10-22", end: "2026-10-28" },
  { week: 3, start: "2026-11-05", end: "2026-11-11" },
  { week: 4, start: "2026-11-12", end: "2026-11-18" },
];
const MONTH_NAMES = ["1월", "2월", "3월", "4월", "5월", "6월",
  "7월", "8월", "9월", "10월", "11월", "12월"];
const WEEKDAY_NAMES = ["일", "월", "화", "수", "목", "금", "토"];

function clonePeriods(periods) {
  return periods.map((p) => ({ week: Number(p.week), start: p.start, end: p.end }));
}

function addDays(ymd, days) {
  const [y, m, d] = ymd.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d + days)).toISOString().slice(0, 10);
}

function parseDay(ymd) {
  const [y, m, d] = ymd.split("-").map(Number);
  if (!y || !m || !d) return null;
  return new Date(Date.UTC(y, m - 1, d));
}

function validYmd(ymd) {
  return /^\d{4}-\d{2}-\d{2}$/.test(ymd) && parseDay(ymd)?.toISOString().slice(0, 10) === ymd;
}

function periodFromStart(week, start) {
  return { week, start, end: validYmd(start) ? addDays(start, 6) : "" };
}

function normalizePayload(payload = {}) {
  payload = payload || {};
  const periods = Array.isArray(payload.periods) && payload.periods.length === 4
    ? payload.periods.map((p, i) => periodFromStart(Number(p.week || i + 1), String(p.start || "")))
    : clonePeriods(DEFAULT_PERIODS);
  return {
    timezone: payload.timezone || TIMEZONE,
    revision: Number.isInteger(payload.revision) ? payload.revision : 0,
    periods,
    current_week: payload.current_week ?? null,
    accepting: payload.accepting !== false,
    demo: Boolean(payload.demo),
    automatic: payload.automatic !== false,
    server_time: payload.server_time,
    applied_to: payload.applied_to || "future_submissions",
  };
}

function formatDate(ymd) {
  if (!validYmd(ymd)) return "";
  const d = parseDay(ymd);
  return `${d.getUTCMonth() + 1}/${d.getUTCDate()}`;
}

function scheduleSummary(schedule) {
  if (schedule.demo) return "Demo는 날짜와 관계없이 1주차 연습";
  if (schedule.current_week) return `${schedule.current_week}주차 접수 중`;
  if (schedule.accepting === false) return "현재 접수 기간 밖";
  return "일정 자동 분류 대기";
}

function collectStarts(inputs) {
  return inputs.map((input) => input.value.trim());
}

function startsToPeriods(starts) {
  return starts.map((start, i) => periodFromStart(i + 1, start));
}

function monthKey(date) {
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
}

function nextMonth(date, step) {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() + step, 1));
}

function firstVisibleMonth(periods) {
  const dates = periods.map((p) => parseDay(p.start)).filter(Boolean).sort((a, b) => a - b);
  return dates[0]
    ? new Date(Date.UTC(dates[0].getUTCFullYear(), dates[0].getUTCMonth(), 1))
    : new Date(Date.UTC(2026, 9, 1));
}

function classForDay(ymd, periods) {
  const day = parseDay(ymd);
  const period = periods.find((p) => {
    const start = parseDay(p.start);
    const end = parseDay(p.end);
    return start && end && day >= start && day <= end;
  });
  if (period) return `w${period.week}`;

  const bounds = periods
    .flatMap((p) => [parseDay(p.start), parseDay(p.end)])
    .filter(Boolean)
    .sort((a, b) => a - b);
  if (bounds.length >= 2 && day > bounds[0] && day < bounds.at(-1)) return "gap";
  return "";
}

function makeNode(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

export function mountSchedule({ state }) {
  const panel = document.getElementById("adminpanel");
  if (!panel || document.getElementById("schedule-pane")) return;

  let schedule = normalizePayload(state.schedule);
  let draftPeriods = clonePeriods(schedule.periods);
  let visibleMonth = firstVisibleMonth(draftPeriods);
  let saving = false;

  const pane = makeNode("div", "apane schedule-pane");
  pane.id = "schedule-pane";
  pane.hidden = true;

  const header = makeNode("div", "schedule-head");
  const titleBox = makeNode("div");
  const title = makeNode("h2", "", "주차 일정");
  const summary = makeNode("p", "hint");
  summary.id = "schedule-summary";
  titleBox.append(title, summary);

  const openButton = makeNode("button", "btn", "주차 일정 설정");
  openButton.type = "button";
  openButton.id = "schedule-open";
  header.append(titleBox, openButton);

  const status = makeNode("p", "hint schedule-status");
  status.id = "schedule-status";
  pane.append(header, status);

  const dialog = document.createElement("dialog");
  dialog.className = "schedule-dialog";
  dialog.id = "schedule-dialog";
  dialog.setAttribute("aria-labelledby", "schedule-dialog-title");

  const form = document.createElement("form");
  form.method = "dialog";
  form.className = "schedule-form";

  const dialogHead = makeNode("div", "schedule-dialog-head");
  const heading = makeNode("h3", "", "주차 일정 설정");
  heading.id = "schedule-dialog-title";
  const closeButton = makeNode("button", "schedule-icon-btn", "×");
  closeButton.type = "button";
  closeButton.id = "schedule-close";
  closeButton.setAttribute("aria-label", "닫기");
  dialogHead.append(heading, closeButton);

  const note = makeNode("p", "hint schedule-note",
    "타임존은 Asia/Seoul 기준입니다. 변경한 일정은 이후 제출되는 의견부터 적용됩니다.");
  const error = makeNode("div", "err schedule-error");
  error.id = "schedule-error";
  error.hidden = true;
  const errorText = makeNode("span");
  error.appendChild(errorText);

  const grid = makeNode("div", "schedule-grid");
  const inputs = [];
  const ends = [];
  for (let i = 1; i <= 4; i += 1) {
    const row = makeNode("div", "schedule-week-row");
    const field = makeNode("div", "field");
    const label = makeNode("label", "", `${i}주차 시작일`);
    label.htmlFor = `schedule-week-${i}`;
    const input = document.createElement("input");
    input.type = "date";
    input.id = `schedule-week-${i}`;
    input.name = `week-${i}`;
    input.required = true;
    input.autocomplete = "off";
    field.append(label, input);

    const endBox = makeNode("div", "schedule-end");
    const endLabel = makeNode("span", "flabel", "종료일");
    const endValue = makeNode("span");
    endValue.id = `schedule-week-${i}-end`;
    endBox.append(endLabel, endValue);
    row.append(field, endBox);
    grid.appendChild(row);
    inputs.push(input);
    ends.push(endValue);
  }

  const previewNav = makeNode("div", "schedule-preview-nav");
  const prev = makeNode("button", "btn", "이전");
  prev.type = "button";
  const monthsLabel = makeNode("span", "schedule-month-label");
  const next = makeNode("button", "btn", "다음");
  next.type = "button";
  previewNav.append(prev, monthsLabel, next);

  const preview = makeNode("div", "schedule-preview");
  preview.id = "schedule-preview";

  const legend = makeNode("div", "schedule-week-legend");
  for (let i = 1; i <= 4; i += 1) {
    const item = makeNode("span");
    const sw = makeNode("i", `w${i}`);
    item.append(sw, document.createTextNode(`${i}주차`));
    legend.appendChild(item);
  }
  const gapItem = makeNode("span");
  gapItem.append(makeNode("i", "gap"), document.createTextNode("미배정 기간"));
  legend.appendChild(gapItem);

  const actions = makeNode("div", "schedule-actions");
  const reloadButton = makeNode("button", "btn", "현재 일정 다시 불러오기");
  reloadButton.type = "button";
  reloadButton.id = "schedule-reload";
  const cancelButton = makeNode("button", "btn", "취소");
  cancelButton.type = "button";
  cancelButton.id = "schedule-cancel";
  const saveButton = makeNode("button", "btn pri", "저장");
  saveButton.type = "submit";
  saveButton.id = "schedule-save";
  actions.append(reloadButton, cancelButton, saveButton);

  form.append(dialogHead, note, error, grid, previewNav, preview, legend, actions);
  dialog.appendChild(form);
  pane.appendChild(dialog);
  panel.prepend(pane);

  function showStatus(message, bad = false) {
    status.textContent = message || "";
    status.style.color = bad ? "var(--bad)" : "";
  }

  function showError(message) {
    error.hidden = !message;
    errorText.textContent = message || "";
  }

  function renderSummary() {
    summary.textContent = scheduleSummary(schedule);
  }

  function syncInputs() {
    inputs.forEach((input, i) => {
      input.value = draftPeriods[i]?.start || "";
      ends[i].textContent = draftPeriods[i]?.end ? `${draftPeriods[i].end} (${formatDate(draftPeriods[i].end)})` : "-";
    });
  }

  function renderMonth(container, monthStart) {
    const month = makeNode("section", "schedule-month");
    const y = monthStart.getUTCFullYear();
    const m = monthStart.getUTCMonth();
    month.appendChild(makeNode("h4", "", `${y}년 ${MONTH_NAMES[m]}`));
    const weekdays = makeNode("div", "schedule-weekdays");
    WEEKDAY_NAMES.forEach((day) => weekdays.appendChild(makeNode("span", "", day)));
    month.appendChild(weekdays);

    const cells = makeNode("div", "schedule-days");
    const first = new Date(Date.UTC(y, m, 1));
    const last = new Date(Date.UTC(y, m + 1, 0)).getUTCDate();
    for (let blank = 0; blank < first.getUTCDay(); blank += 1) {
      cells.appendChild(makeNode("span", "blank"));
    }
    for (let day = 1; day <= last; day += 1) {
      const ymd = new Date(Date.UTC(y, m, day)).toISOString().slice(0, 10);
      const cell = makeNode("span", classForDay(ymd, draftPeriods), String(day));
      cell.title = ymd;
      cells.appendChild(cell);
    }
    month.appendChild(cells);
    container.appendChild(month);
  }

  function renderPreview() {
    preview.replaceChildren();
    renderMonth(preview, visibleMonth);
    renderMonth(preview, nextMonth(visibleMonth, 1));
    monthsLabel.textContent = `${monthKey(visibleMonth)} · ${monthKey(nextMonth(visibleMonth, 1))}`;
  }

  function updateDraftFromInputs() {
    draftPeriods = startsToPeriods(collectStarts(inputs));
    inputs.forEach((input, i) => {
      ends[i].textContent = draftPeriods[i].end ? `${draftPeriods[i].end} (${formatDate(draftPeriods[i].end)})` : "-";
      input.classList.toggle("input-bad", !validYmd(input.value));
    });
    visibleMonth = firstVisibleMonth(draftPeriods);
    renderPreview();
  }

  function applySchedule(nextSchedule, message = "") {
    schedule = normalizePayload(nextSchedule);
    state.schedule = schedule;
    draftPeriods = clonePeriods(schedule.periods);
    visibleMonth = firstVisibleMonth(draftPeriods);
    renderSummary();
    syncInputs();
    renderPreview();
    if (message) showStatus(message);
  }

  async function loadSchedule({ quiet = false } = {}) {
    if (!quiet) showStatus("일정을 불러오는 중…");
    try {
      const res = await fetch(apiUrl("/api/schedule"), { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      applySchedule(await res.json(), quiet ? "" : "현재 일정을 불러왔습니다.");
    } catch {
      applySchedule(schedule, quiet ? "" : "서버 일정을 불러오지 못해 기본 일정을 표시합니다.");
    }
  }

  function validateDraft() {
    const starts = collectStarts(inputs);
    if (starts.some((start) => !validYmd(start))) return "네 주차의 시작일을 모두 날짜로 입력해 주세요.";
    const sorted = startsToPeriods(starts).map((p) => parseDay(p.start));
    for (let i = 1; i < sorted.length; i += 1) {
      if (sorted[i] - sorted[i - 1] < 7 * 86400000)
        return "주차는 시간 순서대로 지정하고, 각 7일 기간이 서로 겹치지 않게 해 주세요.";
    }
    return "";
  }

  async function saveSchedule() {
    const message = validateDraft();
    if (message) {
      showError(message);
      return;
    }
    showError("");
    saving = true;
    saveButton.disabled = true;
    saveButton.textContent = "저장 중…";
    try {
      const res = await fetch(apiUrl("/api/schedule"), {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          code: state.adminCode,
          starts: collectStarts(inputs),
          expected_revision: schedule.revision,
        }),
      });
      const body = await res.json().catch(() => ({}));
      if (res.status === 409) {
        showError("다른 관리자가 먼저 일정을 바꿨습니다. 현재 일정을 다시 불러온 뒤 확인해 주세요.");
        return;
      }
      if (!res.ok) {
        showError(body.message || body.detail || "일정을 저장하지 못했습니다. 입력값과 관리자 권한을 확인해 주세요.");
        return;
      }
      applySchedule(body, "일정을 저장했습니다.");
      document.dispatchEvent(new CustomEvent("atlas:schedule", { detail: schedule }));
      dialog.close();
    } catch {
      showError("네트워크 문제로 일정을 저장하지 못했습니다. 연결을 확인한 뒤 다시 시도해 주세요.");
    } finally {
      saving = false;
      saveButton.disabled = false;
      saveButton.textContent = "저장";
    }
  }

  openButton.addEventListener("click", () => {
    draftPeriods = clonePeriods(schedule.periods);
    visibleMonth = firstVisibleMonth(draftPeriods);
    syncInputs();
    showError("");
    renderPreview();
    dialog.showModal();
    inputs[0].focus();
  });
  closeButton.addEventListener("click", () => dialog.close());
  cancelButton.addEventListener("click", () => dialog.close());
  reloadButton.addEventListener("click", () => loadSchedule());
  prev.addEventListener("click", () => {
    visibleMonth = nextMonth(visibleMonth, -1);
    renderPreview();
  });
  next.addEventListener("click", () => {
    visibleMonth = nextMonth(visibleMonth, 1);
    renderPreview();
  });
  inputs.forEach((input) => input.addEventListener("input", () => {
    showError("");
    updateDraftFromInputs();
  }));
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    saveSchedule();
  });

  document.addEventListener("atlas:hello", (e) => {
    const admin = state.channel === "admin" || e.detail?.channel === "admin";
    pane.hidden = !admin;
    if (admin) {
      if (e.detail?.schedule) applySchedule(e.detail.schedule);
      else loadSchedule({ quiet: true });
    }
  });
  document.addEventListener("atlas:schedule", (e) => {
    if (saving) return;
    const incoming = normalizePayload(e.detail);
    if (incoming.revision < schedule.revision) return;
    const dirty = inputs.some((input, i) => input.value !== schedule.periods[i].start);
    if (dialog.open && dirty) {
      if (incoming.revision > schedule.revision)
        showError("다른 관리자가 일정을 변경했습니다. 작성 중인 날짜는 유지됩니다. 현재 일정을 다시 불러와 확인해 주세요.");
      return;
    }
    applySchedule(incoming);
  });

  applySchedule(schedule);
  renderSummary();
}
