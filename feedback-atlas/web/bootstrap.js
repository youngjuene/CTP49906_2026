const boot = document.documentElement.dataset;

function blockNativeSubmit(event) {
  if (boot.boot === "ready") return;
  event.preventDefault();
}

document.getElementById("admin-form")?.addEventListener("submit", blockNativeSubmit, true);
document.getElementById("boot-retry")?.addEventListener("click", () => location.reload());

try {
  await import("./app.js");
} catch (err) {
  boot.boot = "failed";
  const adminSubmit = document.getElementById("admin-submit");
  if (adminSubmit) {
    adminSubmit.disabled = true;
    adminSubmit.textContent = "시작 실패";
  }
  for (const view of document.querySelectorAll(".view")) view.hidden = view.id !== "view-boot";
  document.getElementById("boot-status").hidden = true;
  document.getElementById("boot-error").hidden = false;
  document.getElementById("boot-error-text").textContent =
    `앱을 시작하지 못했습니다. ${String(err?.message || err)}`;
}
