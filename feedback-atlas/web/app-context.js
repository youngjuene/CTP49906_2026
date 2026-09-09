const scriptBase = new URL(".", import.meta.url);
const basePath = scriptBase.pathname.endsWith("/") ? scriptBase.pathname : `${scriptBase.pathname}/`;

function cleanPath(path) {
  return String(path || "").replace(/^\/+/, "");
}

export function apiUrl(path = "") {
  const url = new URL(cleanPath(path), scriptBase);
  return `${url.pathname}${url.search}${url.hash}`;
}

export function wsUrl(path = "") {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${apiUrl(path)}`;
}

export function storageKey(name) {
  const scope = basePath === "/" ? "classroom" : basePath.replace(/^\/|\/$/g, "").replace(/\//g, ":");
  return `${name}:${scope || "classroom"}`;
}

export async function loadEnvironment() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 4000);
  try {
    const res = await fetch(apiUrl("/api/environment"), {
      cache: "no-store",
      signal: controller.signal,
    });
    if (res.ok) return await res.json();
  } catch {
    /* Older servers do not expose environment metadata. The main classroom
     * remains usable; demo chrome appears when the new endpoint is present. */
  } finally {
    clearTimeout(timeout);
  }
  // A metadata failure must not label a demo URL as the real classroom.
  return basePath === "/demo/"
    ? { mode: "demo", accounts: [], classroom_url: "/" }
    : { mode: "classroom", demo_url: null };
}
