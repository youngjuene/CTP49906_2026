const boot = document.documentElement.dataset;

function blockNativeSubmit(event) {
  if (boot.boot === "ready") return;
  event.preventDefault();
}

document.getElementById("gate-form")?.addEventListener("submit", blockNativeSubmit, true);
document.getElementById("admin-form")?.addEventListener("submit", blockNativeSubmit, true);
document.getElementById("boot-retry")?.addEventListener("click", () => location.reload());

try {
  await import("./app.js");
} catch (err) {
  boot.boot = "failed";
  const error = document.getElementById("gate-error");
  const text = document.getElementById("gate-error-text");
  const retry = document.getElementById("boot-retry");
  const submit = document.getElementById("gate-submit");
  const adminSubmit = document.getElementById("admin-submit");
  if (submit) {
    submit.disabled = true;
    submit.textContent = "시작 실패";
  }
  if (adminSubmit) {
    adminSubmit.disabled = true;
    adminSubmit.textContent = "시작 실패";
  }
  if (error && text && retry) {
    error.hidden = false;
    text.textContent = `앱을 시작하지 못했습니다. ${String(err?.message || err)}`;
    retry.hidden = false;
  }
}
